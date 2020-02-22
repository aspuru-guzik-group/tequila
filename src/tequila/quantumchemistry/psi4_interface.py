from tequila import TequilaException
from openfermion import MolecularData

from tequila.quantumchemistry.qc_base import ParametersQC, QuantumChemistryBase, ClosedShellAmplitudes, Amplitudes

import copy
import numpy
import typing

from dataclasses import dataclass

__HAS_PSI4_PYTHON__ = False
try:
    import psi4

    __HAS_PSI4_PYTHON__ = True
except ModuleNotFoundError:
    __HAS_PSI4_PYTHON__ = False


class TequilaPsi4Exception(TequilaException):
    pass


class OpenVQEEPySCFException(TequilaException):
    pass


@dataclass
class Psi4Results:
    variables: dict = None  # psi4 variables dictionary, storing all computed values
    filename: str = None  # psi4 output file
    wfn: typing.Union[
        psi4.core.Wavefunction, psi4.core.CCWavefunction, psi4.core.CIWavefunction] = None  # psi4 wavefunction
    mol: psi4.core.Molecule = None


class QuantumChemistryPsi4(QuantumChemistryBase):
    @dataclass
    class OrbitalData:
        irrep: str = None
        idx_irrep: int = None
        idx_total: int = None
        energy: float = None

        def __str__(self):
            return "{} : {}{} energy = {:+2.6f} ".format(self.idx_total, self.idx_irrep, self.irrep, self.energy)

    def __init__(self, parameters: ParametersQC, transformation: typing.Union[str, typing.Callable] = None, *args,
                 **kwargs):

        self.energies = {}  # history to avoid recomputation
        self.logs = {}  # store full psi4 output

        super().__init__(parameters=parameters, transformation=transformation, *args, **kwargs)
        self.ref_energy = self.molecule.hf_energy
        self.ref_wfn = self.logs['hf'].wfn
        self.psi4_mol = self.logs['hf'].mol
        self.irreps = [self.psi4_mol.point_group().char_table().gamma(i).symbol() for i in range(self.nirrep)]
        oenergies = []
        for i in self.irreps:
            oenergies += [(i, j, x) for j, x in enumerate(self.orbital_energies(irrep=i))]

        oenergies = sorted(oenergies, key=lambda x: x[2])
        self.orbitals = [self.OrbitalData(irrep=data[0], idx_irrep=data[1], idx_total=i, energy=data[2]) for i, data in
                         enumerate(oenergies)]
        orbitals_by_irrep = {o.irrep: [] for o in self.orbitals}
        for o in self.orbitals:
            orbitals_by_irrep[o.irrep] += [o]

        self.orbitals_by_irrep = orbitals_by_irrep

    @property
    def point_group(self):
        return self.psi4_mol.point_group().symbol()

    @property
    def nirrep(self):
        return self.ref_wfn.nirrep()

    def orbital_energies(self, irrep: typing.Union[int, str] = None, beta: bool = False):
        """
        Returns orbital energies of a given irrep
        or all orbital energies of all irreps (default)

        Parameters
        ----------
        irrep: int or str :
            int: specify the irrep by number (in cotton ordering)
            str: specify the irrep by name (like 'A1')
            specify from which irrep you want the orbital energies
            psi4 orders irreps in 'Cotton ordering'
            http://www.psicode.org/psi4manual/master/psithonmol.html#table-irrepordering
        beta: bool : (Default value=False)
            get the beta electrons

        Returns
        -------
        list or orbital energies
        """

        if hasattr(irrep, "upper"):
            irrep = self.irreps.index(irrep.upper())

        if beta:
            tmp = psi4.driver.p4util.numpy_helper._to_array(self.ref_wfn.epsilon_b(), dense=False)
        else:
            tmp = psi4.driver.p4util.numpy_helper._to_array(self.ref_wfn.epsilon_a(), dense=False)

        if irrep is None:
            result = []
            for x in tmp:
                result += x
            return result
        else:
            return tmp[irrep]

    def make_active_space_hamiltonian(self,
                                      occ: typing.Union[dict, list],
                                      virt: typing.Union[dict, list]):
        """
        Make an active space hamiltonian

        Parameters
        ----------
        occ: dictionary :
            dictionary with irreps as keys and a list of integers as values
            i.e. occ = {"A1":[0,1], "A2":[0]}
            means the occupied active space is made up of spatial orbitals
            0A1, 1A1 and 0A2
            as list: Give a list of spatial orbital indices
            i.e. occ = [0,1,3] means that spatial orbital 0, 1 and 3 are used
        virt
            same format as occ
        Returns
        -------
        Hamiltonian defined in the active space given here

        """
        # transform irrep notation to absolute ints
        occ_idx = occ
        vir_idx = virt
        if isinstance(occ, dict):
            for key, value in occ.items():
                occ_idx = []
                occ_idx += [self.orbitals_by_irrep[key.upper()][i].idx_total for i in value]
        if isinstance(virt, dict):
            for key, value in virt.items():
                vir_idx = []
                vir_idx += [self.orbitals_by_irrep[key.upper()][i].idx_total for i in value]

        print("occ_idx=", occ_idx)
        print("occ_idx=", vir_idx)

        return self.make_hamiltonian(occupied_indices=occ_idx, active_indices=vir_idx)

    def do_make_molecule(self, *args, **kwargs) -> MolecularData:

        energy = self.compute_energy(method="hf", *args, **kwargs)
        wfn = self.logs['hf'].wfn

        molecule = MolecularData(**self.parameters.molecular_data_param)
        if wfn.nirrep() != 1:
            wfn = wfn.c1_deep_copy(wfn.basisset())

        molecule.one_body_integrals = self.compute_one_body_integrals(ref_wfn=wfn)
        molecule.two_body_integrals = self.compute_two_body_integrals(ref_wfn=wfn)
        molecule.hf_energy = energy
        molecule.nuclear_repulsion = wfn.variables()['NUCLEAR REPULSION ENERGY']
        molecule.canonical_orbitals = numpy.asarray(wfn.Ca())
        molecule.overlap_integrals = numpy.asarray(wfn.S())
        molecule.n_orbitals = molecule.canonical_orbitals.shape[0]
        molecule.n_qubits = 2 * molecule.n_orbitals
        molecule.orbital_energies = numpy.asarray(wfn.epsilon_a())
        molecule.fock_matrix = numpy.asarray(wfn.Fa())
        molecule.save()
        return molecule

    def compute_one_body_integrals(self, ref_wfn=None):
        if ref_wfn is None:
            self.compute_energy(method="hf")
            ref_wfn = self.logs['hf'].wfn
        if ref_wfn.nirrep() != 1:
            wfn = ref_wfn.c1_deep_copy(ref_wfn.basisset())
        else:
            wfn = ref_wfn
        Ca = numpy.asarray(wfn.Ca())
        h = wfn.H()
        h = numpy.einsum("xy, yi -> xi", h, Ca, optimize='optimize')
        h = numpy.einsum("xj, xi -> ji", Ca, h, optimize='optimize')
        return h

    def compute_two_body_integrals(self, ref_wfn=None):
        if ref_wfn is None:
            if 'hf' not in self.logs:
                self.compute_energy(method="hf")
            ref_wfn = self.logs['hf'].wfn

        if ref_wfn.nirrep() != 1:
            wfn = ref_wfn.c1_deep_copy(ref_wfn.basisset())
        else:
            wfn = ref_wfn

        mints = psi4.core.MintsHelper(wfn.basisset())

        # Molecular orbitals (coeffs)
        Ca = wfn.Ca()
        h = numpy.asarray(mints.ao_eri())
        h = numpy.einsum("psqr", h, optimize='optimize')  # meet openfermion conventions
        h = numpy.einsum("wxyz, wi -> ixyz", h, Ca, optimize='optimize')
        h = numpy.einsum("wxyz, xi -> wiyz", h, Ca, optimize='optimize')
        h = numpy.einsum("wxyz, yi -> wxiz", h, Ca, optimize='optimize')
        h = numpy.einsum("wxyz, zi -> wxyi", h, Ca, optimize='optimize')
        return h

    def compute_ccsd_amplitudes(self):
        return self.compute_amplitudes(method='ccsd')

    def _run_psi4(self, options: dict = None, method=None, return_wfn=True, point_group=None, filename: str = None,
                  guess_wfn=None, *args, **kwargs):
        psi4.core.clean()

        if "threads" in kwargs:
            psi4.set_num_threads(nthread=kwargs["threads"])

        if filename is None:
            filename = "{}_{}.out".format(self.parameters.filename, method)

        psi4.core.set_output_file(filename)

        defaults = {'basis': self.parameters.basis_set,
                    'e_convergence': 1e-8,
                    'd_convergence': 1e-8}
        if options is None:
            options = {}
        options = {**defaults, **options}

        # easier guess read in
        if guess_wfn is not None:
            if isinstance(guess_wfn, QuantumChemistryPsi4):
                guess_wfn = guess_wfn.logs["hf"].wfn
            if isinstance(guess_wfn, str):
                guess_wfn = psi4.core.Wavefunction.from_file(guess_wfn)
            guess_wfn.to_file(guess_wfn.get_scratch_filename(180))  # this is necessary
            options["guess"] = "read"

        # prevent known flaws
        if "guess" in options and options["guess"].lower() == "read":
            options["basis_guess"] = False
            # additionally the outputfile needs to be the same
            # as in the previous guess
            # this can not be determined here
            # better pass down a guess_wfn

        psi4.set_options(options)

        mol = psi4.geometry(self.parameters.get_geometry_string())

        if point_group is not None:
            mol.reset_point_group(point_group.lower())

        energy, wfn = psi4.energy(name=method.lower(), return_wfn=return_wfn, molecule=mol, guess_wfn=guess_wfn)
        self.energies[method.lower()] = energy
        self.logs[method.lower()] = Psi4Results(filename=filename, variables=copy.deepcopy(psi4.core.variables()),
                                                wfn=wfn, mol=mol)

        return energy, wfn

    def compute_amplitudes(self, method: str, options: dict = None, filename: str = None) -> typing.Union[
        Amplitudes, ClosedShellAmplitudes]:
        if __HAS_PSI4_PYTHON__:
            energy, wfn = self._run_psi4(method=method, options=options, point_group='c1', filename=filename)
            all_amplitudes = wfn.get_amplitudes()
            closed_shell = isinstance(wfn.reference_wavefunction(), psi4.core.RHF)
            if closed_shell:
                return ClosedShellAmplitudes(**{k: v.to_array() for k, v in all_amplitudes.items()})
            else:
                return Amplitudes(**{k: v.to_array() for k, v in all_amplitudes.items()})

        else:
            raise TequilaPsi4Exception("Can't find the psi4 python module, let your environment know the path to psi4")

    def compute_energy(self, method: str = "fci", options=None, *args, **kwargs):
        if method.lower() in self.energies:
            return self.energies[method.lower()]
        if __HAS_PSI4_PYTHON__:
            return self._run_psi4(method=method, options=options, *args, **kwargs)[0]
        else:
            raise TequilaPsi4Exception("Can't find the psi4 python module, let your environment know the path to psi4")

    def __str__(self):
        result = super().__str__()
        result += "\nPsi4 Data\n"
        result += "{key:15} : {value:15} \n".format(key="Point Group (full)",
                                                    value=self.psi4_mol.get_full_point_group().lower())
        result += "{key:15} : {value:15} \n".format(key="Point Group (used)", value=self.point_group)
        result += "{key:15} : {value} \n".format(key="nirrep", value=self.nirrep)
        result += "{key:15} : {value} \n".format(key="irreps", value=self.irreps)
        result += "{key:15} : {value:15} \n".format(key="mos per irrep", value=str(
            [len(self.orbital_energies(irrep=i)) for i in range(self.nirrep)]))
        return result
