from dataclasses import dataclass
from tequila import TequilaException, BitString, TequilaWarning
from tequila.hamiltonian import QubitHamiltonian
from tequila.wavefunction import QubitWaveFunction
from tequila.hamiltonian.paulis import Sp, Sm, Qp, Qm

from tequila.circuit import QCircuit, gates, _gates_impl
from tequila.objective.objective import Variable, Variables, ExpectationValue

from tequila.simulators.simulator_api import simulate
from tequila.utils import to_float

from tequila.objective import assign_variable

from .encodings import known_encodings

import typing, numpy, numbers, copy
from itertools import product

# if you are experiencing import errors you need to update openfermion
# required is version >= 1.0
# otherwise replace with from openfermion.hamiltonians import MolecularData
import openfermion
from openfermion.chem import MolecularData

import warnings


@dataclass
class ActiveSpaceData:
    active_orbitals: list  # active orbitals (spatial, c1)
    reference_orbitals: list  # reference orbitals (spatial, c1)

    def __str__(self):
        result = "Active Space Data:\n"
        result += "{key:15} : {value:15} \n".format(key="active_orbitals", value=str(self.active_orbitals))
        result += "{key:15} : {value:15} \n".format(key="reference_orbitals",
                                                    value=str(self.reference_orbitals))
        result += "{key:15} : {value:15} \n".format(key="frozen_docc", value=str(self.frozen_docc))
        result += "{key:15} : {value:15} \n".format(key="frozen_uocc", value=str(self.frozen_uocc))
        return result

    @property
    def frozen_reference_orbitals(self):
        return [i for i in self.reference_orbitals if i not in self.active_orbitals]

    @property
    def active_reference_orbitals(self):
        return [i for i in self.reference_orbitals if i in self.active_orbitals]


class FermionicGateImpl(gates.QubitExcitationImpl):
    # keep the overview in circuits
    def __init__(self, generator, p0, transformation,  *args, **kwargs):
        super().__init__(generator=generator, p0=p0, *args, **kwargs)
        self._name = "FermionicExcitation"
        self.transformation=transformation

    def compile(self):
        return gates.Trotterized(generator=self.generator, control=self.control, angle=self.parameter, steps=1)

def prepare_product_state(state: BitString) -> QCircuit:
    """Small convenience function

    Parameters
    ----------
    state :
        product state encoded into a bitstring
    state: BitString :


    Returns
    -------
    type
        unitary circuit which prepares the product state

    """
    result = QCircuit()
    for i, v in enumerate(state.array):
        if v == 1:
            result += gates.X(target=i)
    return result


@dataclass
class ParametersQC:
    """Specialization of ParametersHamiltonian"""
    basis_set: str = ''  # Quantum chemistry basis set
    geometry: str = ''  # geometry of the underlying molecule (units: Angstrom!),
    # this can be a filename leading to an .xyz file or the geometry given as a string
    description: str = ''
    multiplicity: int = 1
    charge: int = 0
    closed_shell: bool = True
    name: str = "molecule"

    @property
    def filename(self):
        """ """
        return "{}_{}".format(self.name, self.basis_set)

    @property
    def molecular_data_param(self) -> dict:
        """:return: Give back all parameters for the MolecularData format from openfermion as dictionary"""
        return {'basis': self.basis_set, 'geometry': self.get_geometry(), 'description': self.description,
                'charge': self.charge, 'multiplicity': self.multiplicity, 'filename': self.filename
                }

    @staticmethod
    def format_element_name(string):
        """OpenFermion uses case sensitive hash tables for chemical elements
        I.e. you need to name Lithium: 'Li' and 'li' or 'LI' will not work
        this convenience function does the naming
        :return: first letter converted to upper rest to lower

        Parameters
        ----------
        string :


        Returns
        -------

        """
        assert (len(string) > 0)
        assert (isinstance(string, str))
        fstring = string[0].upper() + string[1:].lower()
        return fstring

    @staticmethod
    def convert_to_list(geometry):
        """Convert a molecular structure given as a string into a list suitable for openfermion

        Parameters
        ----------
        geometry :
            a string specifying a mol. structure. E.g. geometry="h 0.0 0.0 0.0\n h 0.0 0.0 1.0"

        Returns
        -------
        type
            A list with the correct format for openfermion E.g return [ ['h',[0.0,0.0,0.0], [..]]

        """
        result = []
        for line in geometry.split('\n'):
            words = line.split()
            if len(words) != 4:  break
            try:
                tmp = (ParametersQC.format_element_name(words[0]),
                       (float(words[1]), float(words[2]), float(words[3])))
                result.append(tmp)
            except ValueError:
                print("get_geometry list unknown line:\n ", line, "\n proceed with caution!")
        return result

    def get_geometry_string(self) -> str:
        """returns the geometry as a string
        :return: geometry string

        Parameters
        ----------

        Returns
        -------

        """
        if self.geometry.split('.')[-1] == 'xyz':
            geomstring, comment = self.read_xyz_from_file(self.geometry)
            if comment is not None:
                self.description = comment
            return geomstring
        else:
            return self.geometry

    def get_geometry(self):
        """Returns the geometry
        If a xyz filename was given the file is read out
        otherwise it is assumed that the geometry was given as string
        which is then reformatted as a list usable as input for openfermion
        :return: geometry as list
        e.g. [(h,(0.0,0.0,0.35)),(h,(0.0,0.0,-0.35))]
        Units: Angstrom!

        Parameters
        ----------

        Returns
        -------

        """
        if self.geometry.split('.')[-1] == 'xyz':
            geomstring, comment = self.read_xyz_from_file(self.geometry)
            if self.description == '':
                self.description = comment
            if self.name == "molecule":
                self.name = self.geometry.split('.')[0]
            return self.convert_to_list(geomstring)
        elif self.geometry is not None:
            return self.convert_to_list(self.geometry)
        else:
            raise Exception("Parameters.qc.geometry is None")

    @staticmethod
    def read_xyz_from_file(filename):
        """Read XYZ filetype for molecular structures
        https://en.wikipedia.org/wiki/XYZ_file_format
        Units: Angstrom!

        Parameters
        ----------
        filename :
            return:

        Returns
        -------

        """
        with open(filename, 'r') as file:
            content = file.readlines()
            natoms = int(content[0])
            comment = str(content[1]).strip('\n')
            coord = ''
            for i in range(natoms):
                coord += content[2 + i]
            return coord, comment


@dataclass
class ClosedShellAmplitudes:
    """ """
    tIjAb: numpy.ndarray = None
    tIA: numpy.ndarray = None

    def make_parameter_dictionary(self, threshold=1.e-8):
        """

        Parameters
        ----------
        threshold :
             (Default value = 1.e-8)

        Returns
        -------

        """
        variables = {}
        if self.tIjAb is not None:
            nvirt = self.tIjAb.shape[2]
            nocc = self.tIjAb.shape[0]
            assert (self.tIjAb.shape[1] == nocc and self.tIjAb.shape[3] == nvirt)
            for (I, J, A, B), value in numpy.ndenumerate(self.tIjAb):
                if not numpy.isclose(value, 0.0, atol=threshold):
                    variables[(nocc + A, I, nocc + B, J)] = value
        if self.tIA is not None:
            nocc = self.tIA.shape[0]
            for (I, A), value, in numpy.ndenumerate(self.tIA):
                if not numpy.isclose(value, 0.0, atol=threshold):
                    variables[(A + nocc, I)] = value

        return dict(sorted(variables.items(), key=lambda x: numpy.abs(x[1]), reverse=True))


@dataclass
class Amplitudes:
    """Coupled-Cluster Amplitudes
    We adopt the Psi4 notation for consistency
    I,A for alpha
    i,a for beta

    Parameters
    ----------

    Returns
    -------

    """

    @classmethod
    def from_closed_shell(cls, cs: ClosedShellAmplitudes):
        """
        Initialize from closed-shell Amplitude structure

        Parameters
        ----------
        cs: ClosedShellAmplitudes :


        Returns
        -------

        """
        tijab = cs.tIjAb - numpy.einsum("ijab -> ijba", cs.tIjAb, optimize='greedy')
        return cls(tIjAb=cs.tIjAb, tIA=cs.tIA, tiJaB=cs.tIjAb, tia=cs.tIA, tijab=tijab, tIJAB=tijab)

    tIjAb: numpy.ndarray = None
    tIA: numpy.ndarray = None
    tiJaB: numpy.ndarray = None
    tijab: numpy.ndarray = None
    tIJAB: numpy.ndarray = None
    tia: numpy.ndarray = None

    def make_parameter_dictionary(self, threshold=1.e-8):
        """

        Parameters
        ----------
        threshold :
             (Default value = 1.e-8)
             Neglect amplitudes below the threshold

        Returns
        -------
        Dictionary of tequila variables (hash is in the style of (a,i,b,j))

        """
        variables = {}
        if self.tIjAb is not None:
            nvirt = self.tIjAb.shape[2]
            nocc = self.tIjAb.shape[0]
            assert (self.tIjAb.shape[1] == nocc and self.tIjAb.shape[3] == nvirt)

            for (I, j, A, b), value in numpy.ndenumerate(self.tIjAb):
                if not numpy.isclose(value, 0.0, atol=threshold):
                    variables[(2 * (nocc + A), 2 * I, 2 * (nocc + b) + 1, j + 1)] = value
            for (i, J, a, B), value in numpy.ndenumerate(self.tiJaB):
                if not numpy.isclose(value, 0.0, atol=threshold):
                    variables[(2 * (nocc + a) + 1, 2 * i + 1, 2 * (nocc + B), J)] = value
            for (i, j, a, b), value in numpy.ndenumerate(self.tijab):
                if not numpy.isclose(value, 0.0, atol=threshold):
                    variables[(2 * (nocc + a) + 1, 2 * i + 1, 2 * (nocc + b) + 1, j + 1)] = value
            for (I, J, A, B), value in numpy.ndenumerate(self.tijab):
                if not numpy.isclose(value, 0.0, atol=threshold):
                    variables[(2 * (nocc + A), 2 * I, 2 * (nocc + B), J)] = value

        if self.tIA is not None:
            nocc = self.tIjAb.shape[0]
            assert (self.tia.shape[0] == nocc)
            for (I, A), value, in numpy.ndenumerate(self.tIA):
                if not numpy.isclose(value, 0.0, atol=threshold):
                    variables[(2 * (A + nocc), 2 * I)] = value
            for (i, a), value, in numpy.ndenumerate(self.tIA):
                if not numpy.isclose(value, 0.0, atol=threshold):
                    variables[(2 * (a + nocc) + 1, 2 * i + 1)] = value

        return variables


class NBodyTensor:
    """ Convenience class for handling N-body tensors """

    class Ordering:
        def __init__(self, scheme):
            if hasattr(scheme, "_scheme"):
                scheme = scheme._scheme
            elif hasattr(scheme, "scheme"):
                scheme = scheme.scheme
            self._scheme = self.assign_scheme(scheme)

        def assign_scheme(self, scheme):
            if scheme is None:
                return "chem"
            else:
                scheme = str(scheme)

            if scheme.lower() in ["mulliken", "chem", "c", "1122"]:
                return "chem"
            elif scheme.lower() in ["dirac", "phys", "p", "1212"]:
                return "phys"
            elif scheme.lower() in ["openfermion", "of", "o", "1221"]:
                return "of"
            else:
                raise TequilaException(
                    "Unknown two-body tensor scheme {}. Supported are dirac, mulliken, and openfermion".format(scheme))

        def is_phys(self):
            return self._scheme == "phys"

        def is_chem(self):
            return self._scheme == "chem"

        def is_of(self):
            return self._scheme == "of"

    def __init__(self, elems: numpy.ndarray = None, active_indices: list = None, ordering: str = None,
                 size_full: int = None):
        """
        Parameters
        ----------
        elems: Tensor data as numpy array
        active_indices: List of active indices in total ordering
        ordering: Ordering scheme for two body tensors
        "dirac" or "phys": <12|g|12>
            .. math::
                g_{pqrs} = \\int d1 d2 p(1)q(2) g(1,2) r(1)s(2)
        "mulliken" or "chem": (11|g|22)
            .. math::
                g_{pqrs} = \\int d1 d2 p(1)r(2) g(1,2) q(1)s(2)
        "openfermion":
            .. math:: [12|g|21]
                g_{gqprs} = \\int d1 d2 p(1)q(2) g(1,2) s(1)r(2)

        size_full
        """

        # Set elements
        self.elems = elems
        # Active indices only as list of indices (e.g. spatial orbital indices), not as a dictionary of irreducible
        # representations
        if active_indices is not None:
            self.active_indices = active_indices
        self._passive_indices = None
        self._full_indices = None
        self._indices_set: bool = False

        # Determine order of tensor
        # Assume, that tensor is entered in desired shape, not as flat array.
        self.order = len(self.elems.shape)
        # Can use size_full < self.elems.shape[0] -> 'full' space is to be considered a subspace as well
        if size_full is None:
            self._size_full = self.elems.shape[0]
        else:
            self._size_full = size_full
        # 2-body tensors (<=> order 4) currently allow reordering
        if self.order == 4:
            self.ordering = self.Ordering(ordering)
        else:
            if ordering is not None:
                raise Exception("Ordering only implemented for tensors of order 4 / 2-body tensors.")
            self.ordering = None

    def sub_lists(self, idx_lists: list = None) -> numpy.ndarray:
        """
        Get subspace of tensor by a set of index lists
        according to hPQ.sub_lists(idx_lists=[p, q]) = [hPQ for P in p and Q in q]

        This essentially is an implementation of a non-contiguous slicing using numpy.take

        Parameters
        ----------
            idx_lists :
                List of lists, each defining the desired subspace per axis
                Size needs to match order of tensor, and lists successively correspond to axis=0,1,2,...,N

        Returns
        -------
            out :
                Sliced tensor as numpy.ndarray
        """
        # Check if index list has correct size
        if len(idx_lists) != self.order:
            raise Exception("Need to pass an index list for each dimension!" +
                            " Length of idx_lists needs to match order of tensor.")

        # Perform slicing via numpy.take
        out = self.elems
        for ax in range(self.order):
            if idx_lists[ax] is not None:  # None means, we want the full space in this direction
                out = numpy.take(out, idx_lists[ax], axis=ax)

        return out

    def set_index_lists(self):
        """ Set passive and full index lists based on class inputs """
        tmp_size = self._size_full
        if self._size_full is None:
            tmp_size = self.elems.shape[0]

        self._passive_indices = [i for i in range(tmp_size)
                                 if i not in self.active_indices]
        self._full_indices = [i for i in range(tmp_size)]

    def sub_str(self, name: str) -> numpy.ndarray:
        """
        Get subspace of tensor by a string
        Currently is able to resolve an active space, named 'a', full space 'f', and the complement 'p' = 'f' - 'a'.
        Full space in this context may also be smaller than actual tensor dimension.

        The specification of active space in this context only allows to pick a set from a list of orbitals, and
        is not able to resolve an active space from irreducible representations.

        Example for one-body tensor:
        hPQ.sub_lists(name='ap') = [hPQ for P in active_indices and Q in _passive_indices]

        Parameters
        ----------
            name :
                String specifying the desired subspace, elements need to be a (active), f (full), p (full - active)

        Returns
        -------
            out :
                Sliced tensor as numpy.ndarray
        """
        if not self._indices_set:
            self.set_index_lists()
            self._indices_set = True

        if name is None:
            raise Exception("No name specified.")
        if len(name) != self.order:
            raise Exception("Name does not match order of the tensor.")
        if self.active_indices is None:
            raise Exception("Need to set an active space in order to call this function.")

        idx_lists = []
        # Parse name as string of space indices
        for char in name:
            if char.lower() == 'a':
                idx_lists.append(self.active_indices)
            elif char.lower() == 'p':
                idx_lists.append(self._passive_indices)
            elif char.lower() == 'f':
                if self._size_full is None:
                    idx_lists.append(None)
                else:
                    idx_lists.append(self._full_indices)
            else:
                raise Exception("Need to specify a valid letter (a,p,f).")

        out = self.sub_lists(idx_lists)

        return out

    def reorder(self, to: str = 'of'):
        """
        Function to reorder tensors according to some convention.

        Parameters
        ----------
        to :
            Ordering scheme of choice.
            'openfermion', 'of' (default) :
                openfermion - ordering, corresponds to integrals of the type
                h^pq_rs = int p(1)* q(2)* O(1,2) r(2) s(1) (O(1,2)
                with operators a^pq_rs = a^p a^q a_r a_s (a^p == a^dagger_p)
                currently needed for dependencies on openfermion-library
            'chem', 'c' :
                quantum chemistry ordering, collect particle terms,
                more convenient for real-space methods
                h^pq_rs = int p(1) q(1) O(1,2) r(2) s(2)
                This is output by psi4
            'phys', 'p' :
                typical physics ordering, integrals of type
                h^pq_rs = int p(1)* q(2)* O(1,2) r(1) s(2)
                with operators a^pq_rs = a^p a^q a_s a_r

            Returns
            -------
        """
        if self.order != 4:
            raise Exception('Reordering currently only implemented for two-body tensors.')

        to = self.Ordering(to)

        if self.ordering == to:
            return self
        elif self.ordering.is_chem():
            if to.is_of():
                self.elems = numpy.einsum("psqr -> pqrs", self.elems, optimize='greedy')
            elif to.is_phys():
                self.elems = numpy.einsum("prqs -> pqrs", self.elems, optimize='greedy')
        elif self.ordering.is_of():
            if to.is_chem():
                self.elems = numpy.einsum("pqrs -> psqr", self.elems, optimize='greedy')
            elif to.is_phys():
                self.elems = numpy.einsum("pqrs -> pqsr", self.elems, optimize='greedy')
        elif self.ordering.is_phys():
            if to.is_chem():
                self.elems = numpy.einsum("pqrs -> prqs", self.elems, optimize='greedy')
            elif to.is_of():
                self.elems = numpy.einsum("pqsr -> pqrs", self.elems, optimize='greedy')

        return self

class QuantumChemistryBase:

    def __init__(self, parameters: ParametersQC,
                 transformation: typing.Union[str, typing.Callable] = None,
                 active_orbitals: list = None,
                 *args,
                 **kwargs):

        self.parameters = parameters

        if "molecule" in kwargs:
            self.molecule = kwargs["molecule"]
        else:
            self.molecule = self.make_molecule(*args, **kwargs)

        assert (parameters.basis_set.lower() == self.molecule.basis.lower())
        assert (parameters.multiplicity == self.molecule.multiplicity)
        assert (parameters.charge == self.molecule.charge)

        self.active_space = None
        if active_orbitals is not None:
            self.active_space = self._make_active_space_data(active_orbitals=active_orbitals)

        self.transformation = self._initialize_transformation(transformation=transformation, *args, **kwargs)

        self._rdm1 = None
        self._rdm2 = None

    def _initialize_transformation(self, transformation=None, *args, **kwargs):

        if transformation is None:
            transformation = "JordanWigner"

        # filter out arguments to the transformation
        trafo_args = {k.split("__")[1]: v for k, v in kwargs.items() if
                      (hasattr(k, "lower") and "transformation__" in k.lower())}

        trafo_args["n_electrons"] = self.n_electrons
        trafo_args["n_orbitals"] = self.n_orbitals

        if hasattr(transformation, "upper"):
            # format to conventions
            transformation = transformation.replace("_", "").replace("-", "").upper()
            encodings = known_encodings()
            if transformation in encodings:
                transformation = encodings[transformation](**trafo_args)
            else:
                raise TequilaException(
                    "Unkown Fermion-to-Qubit encoding {}. Try something like: {}".format(transformation,
                                                                                         list(encodings.keys())))

        return transformation

    def _make_active_space_data(self, active_orbitals, reference=None):
        """
        Small helper function
        Internal use only
        Parameters
        ----------
        active_orbitals: dictionary :
            list: Give a list of spatial orbital indices
            i.e. occ = [0,1,3] means that spatial orbital 0, 1 and 3 are used
        reference: (Default value=None)
            List of orbitals which form the reference
            Can be given in the same format as active_orbitals
            If given as None then the first N_electron/2 orbitals are taken
            for closed-shell systems.

        Returns
        -------
        Dataclass with active indices and reference indices (in spatial notation)

        """

        if active_orbitals is None:
            return None

        if reference is None:
            # auto assignment only for closed-shell
            assert (self.n_electrons % 2 == 0)
            reference = sorted([i for i in range(self.n_electrons // 2)])

        return ActiveSpaceData(active_orbitals=sorted(active_orbitals),
                               reference_orbitals=sorted(reference))

    @classmethod
    def from_openfermion(cls, molecule: openfermion.MolecularData,
                         transformation: typing.Union[str, typing.Callable] = None,
                         *args,
                         **kwargs):
        """
        Initialize direclty from openfermion MolecularData object

        Parameters
        ----------
        molecule
            The openfermion molecule
        Returns
        -------
            The Tequila molecule
        """
        parameters = ParametersQC(basis_set=molecule.basis, geometry=molecule.geometry,
                                  description=molecule.description, multiplicity=molecule.multiplicity,
                                  charge=molecule.charge)
        return cls(parameters=parameters, transformation=transformation, molecule=molecule, *args, **kwargs)

    def make_excitation_generator(self,
                                  indices: typing.Iterable[typing.Tuple[int, int]],
                                  form: str = None,
                                  remove_constant_term: bool = True) -> QubitHamiltonian:
        """
        Notes
        ----------
        Creates the transformed hermitian generator of UCC type unitaries:
              M(a^\dagger_{a_0} a_{i_0} a^\dagger{a_1}a_{i_1} ... - h.c.)
              where the qubit map M depends is self.transformation

        Parameters
        ----------
        indices : typing.Iterable[typing.Tuple[int, int]] :
            List of tuples [(a_0, i_0), (a_1, i_1), ... ] - recommended format, in spin-orbital notation (alpha odd numbers, beta even numbers)
            can also be given as one big list: [a_0, i_0, a_1, i_1 ...]
        form : str : (Default value None):
            Manipulate the generator to involution or projector
            set form='involution' or 'projector'
            the default is no manipulation which gives the standard fermionic excitation operator back
        remove_constant_term: bool: (Default value True):
            by default the constant term in the qubit operator is removed since it has no effect on the unitary it generates
            if the unitary is controlled this might not be true!
        Returns
        -------
        type
            1j*Transformed qubit excitation operator, depends on self.transformation
        """

        if type(self.transformation).__name__ == "BravyiKitaevFast":
            raise TequilaException(
                "The Bravyi-Kitaev-Superfast transformation does not support general FermionOperators yet")

        # check indices and convert to list of tuples if necessary
        if len(indices) == 0:
            raise TequilaException("make_excitation_operator: no indices given")
        elif not isinstance(indices[0], typing.Iterable):
            if len(indices) % 2 != 0:
                raise TequilaException("make_excitation_generator: unexpected input format of indices\n"
                                       "use list of tuples as [(a_0, i_0),(a_1, i_1) ...]\n"
                                       "or list as [a_0, i_0, a_1, i_1, ... ]\n"
                                       "you gave: {}".format(indices))
            converted = [(indices[2 * i], indices[2 * i + 1]) for i in range(len(indices) // 2)]
        else:
            converted = indices

        # convert everything to native python int
        # otherwise openfermion will complain
        converted = [(int(pair[0]), int(pair[1])) for pair in converted]

        # convert to openfermion input format
        ofi = []
        dag = []
        for pair in converted:
            assert (len(pair) == 2)
            ofi += [(int(pair[0]), 1),
                    (int(pair[1]), 0)]  # openfermion does not take other types of integers like numpy.int64
            dag += [(int(pair[0]), 0), (int(pair[1]), 1)]

        op = openfermion.FermionOperator(tuple(ofi), 1.j)  # 1j makes it hermitian
        op += openfermion.FermionOperator(tuple(reversed(dag)), -1.j)

        if isinstance(form, str) and form.lower() != 'fermionic':
            # indices for all the Na operators
            Na = [x for pair in converted for x in [(pair[0], 1), (pair[0], 0)]]
            # indices for all the Ma operators (Ma = 1 - Na)
            Ma = [x for pair in converted for x in [(pair[0], 0), (pair[0], 1)]]
            # indices for all the Ni operators
            Ni = [x for pair in converted for x in [(pair[1], 1), (pair[1], 0)]]
            # indices for all the Mi operators
            Mi = [x for pair in converted for x in [(pair[1], 0), (pair[1], 1)]]

            # can gaussianize as projector or as involution (last is default)
            if form.lower() == "p+":
                op *= 0.5
                op += openfermion.FermionOperator(Na + Mi, 0.5)
                op += openfermion.FermionOperator(Ni + Ma, 0.5)
            elif form.lower() == "p-":
                op *= 0.5
                op += openfermion.FermionOperator(Na + Mi, -0.5)
                op += openfermion.FermionOperator(Ni + Ma, -0.5)

            elif form.lower() == "g+":
                op += openfermion.FermionOperator([], 1.0)  # Just for clarity will be subtracted anyway
                op += openfermion.FermionOperator(Na + Mi, -1.0)
                op += openfermion.FermionOperator(Ni + Ma, -1.0)
            elif form.lower() == "g-":
                op += openfermion.FermionOperator([], -1.0)  # Just for clarity will be subtracted anyway
                op += openfermion.FermionOperator(Na + Mi, 1.0)
                op += openfermion.FermionOperator(Ni + Ma, 1.0)
            elif form.lower() == "p0":
                # P0: we only construct P0 and don't keep the original generator
                op = openfermion.FermionOperator([], 1.0)  # Just for clarity will be subtracted anyway
                op += openfermion.FermionOperator(Na + Mi, -1.0)
                op += openfermion.FermionOperator(Ni + Ma, -1.0)
            else:
                raise TequilaException(
                    "Unknown generator form {}, supported are G, P+, P-, G+, G- and P0".format(form))

        qop = self.transformation(op)

        # remove constant terms
        # they have no effect in the unitary (if not controlled)
        if remove_constant_term:
            qop.qubit_operator.terms[tuple()] = 0.0

        # check if the operator is hermitian and cast coefficients to floats
        # in order to avoid trouble with the simulation backends
        assert qop.is_hermitian()
        for k, v in qop.qubit_operator.terms.items():
            qop.qubit_operator.terms[k] = to_float(v)

        qop = qop.simplify()

        if len(qop) == 0:
            warnings.warn("Excitation generator is a unit operator.\n"
                          "Non-standard transformations might not work with general fermionic operators\n"
                          "indices = " + str(indices), category=TequilaWarning)
        return qop

    def make_hardcore_boson_excitation_gate(self, indices, angle, control=None, assume_real=True):
        target = []
        for pair in indices:
            assert len(pair) == 2
            target += [pair[0], pair[1]]
        consistency = [x < self.n_orbitals for x in target]
        if not all(consistency):
            raise TequilaException(
                "make_hardcore_boson_excitation_gate: Inconsistencies in indices={}. Should be indexed from 0 ... n_orbitals={}".format(
                    indices, self.n_orbitals))
        return gates.QubitExcitation(angle=angle, target=target, assume_real=assume_real, control=control)

    def make_excitation_gate(self, indices, angle, control=None, assume_real=True):
        """
        Initialize a fermionic excitation gate defined as

        .. math::
            e^{-i\\frac{a}{2} G}
        with generator defines by the indices [(p0,q0),(p1,q1),...]
        .. math::
            G = i(\\prod_{k} a_{p_k}^\\dagger a_{q_k} - h.c.)

        Parameters
        ----------
            indices:
                List of tuples that define the generator
            angle:
                Numeric or hashable type or tequila objective
            control:
                List of possible control qubits
            assume_real:
                Assume that the wavefunction will always stay real.
                Will reduce potential gradient costs by a factor of 2
        """
        generator = self.make_excitation_generator(indices=indices, remove_constant_term=control is None)
        p0 = self.make_excitation_generator(indices=indices, form="P0", remove_constant_term=control is None)

        return QCircuit.wrap_gate(
            FermionicGateImpl(angle=angle, generator=generator, p0=p0, transformation=type(self.transformation).__name__.lower(), assume_real=assume_real, control=control))

    def make_molecule(self, *args, **kwargs) -> MolecularData:
        """Creates a molecule in openfermion format by running psi4 and extracting the data
        Will check for previous outputfiles before running
        Will not recompute if a file was found

        Parameters
        ----------
        parameters :
            An instance of ParametersQC, which also holds an instance of ParametersPsi4 via parameters.psi4
            The molecule will be saved in parameters.filename, if this file exists before the call the molecule will be imported from the file

        Returns
        -------
        type
            the molecule in openfermion.MolecularData format

        """
        molecule = MolecularData(**self.parameters.molecular_data_param)
        # try to load

        do_compute = True
        try:
            import os
            if os.path.exists(self.parameters.filename):
                molecule.load()
                do_compute = False
        except OSError:
            do_compute = True

        if do_compute:
            molecule = self.do_make_molecule(*args, **kwargs)

        molecule.save()
        return molecule

    def do_make_molecule(self, *args, **kwargs):
        """

        Parameters
        ----------
        args
        kwargs

        Returns
        -------

        """
        # integrals need to be passed in base class
        assert ("one_body_integrals" in kwargs)
        assert ("two_body_integrals" in kwargs)
        one_body_integrals = kwargs["one_body_integrals"]
        two_body_integrals = kwargs["two_body_integrals"]
        if "nuclear_repulsion" in kwargs:
            nuclear_repulsion = kwargs["nuclear_repulsion"]
        else:
            nuclear_repulsion = 0.0
            warnings.warn("No nuclear_repulsion given for custom molecule, setting to zero", category=TequilaWarning)

        if ("n_orbitals" in kwargs):
            n_orbitals = kwargs["n_orbitals"]
        else:
            n_orbitals = one_body_integrals.shape[0]
            for i in [0, 1, 2, 3]:
                assert n_orbitals == two_body_integrals.shape[i]

        molecule = MolecularData(**self.parameters.molecular_data_param)

        molecule.one_body_integrals = one_body_integrals
        molecule.two_body_integrals = two_body_integrals
        molecule.nuclear_repulsion = nuclear_repulsion
        molecule.n_orbitals = n_orbitals
        if "n_electrons" in kwargs:
            molecule.n_electrons = kwargs["n_electrons"]
        molecule.save()
        return molecule

    @property
    def n_orbitals(self) -> int:
        """ """
        if self.active_space is None:
            return self.molecule.n_orbitals
        else:
            return len(self.active_space.active_orbitals)

    @property
    def n_electrons(self) -> int:
        """ """
        if self.active_space is None:
            return self.molecule.n_electrons
        else:
            return 2 * len(self.active_space.active_reference_orbitals)

    def make_hamiltonian(self, occupied_indices=None, active_indices=None, threshold=1.e-8) -> QubitHamiltonian:
        """ """
        if occupied_indices is None and self.active_space is not None:
            occupied_indices = self.active_space.frozen_reference_orbitals
        if active_indices is None and self.active_space is not None:
            active_indices = self.active_space.active_orbitals

        fop = openfermion.transforms.get_fermion_operator(
            self.molecule.get_molecular_hamiltonian(occupied_indices, active_indices))
        try:
            qop = self.transformation(fop)
        except TypeError:
            qop = self.transformation(openfermion.transforms.get_interaction_operator(fop))
        qop.is_hermitian()
        return qop

    def make_hardcore_boson_hamiltonian(self):
        if not self.transformation.up_then_down:
            warnings.warn(
                "Hardcore-Boson Hamiltonian without reordering will result in non-consecutive Hamiltonians that are eventually not be combinable with other features of tequila. Try transformation=\'ReorderedJordanWigner\' or similar for more consistency",
                TequilaWarning)
        # integrate with QubitEncoding at some point
        n_orbitals = self.n_orbitals
        c, obt, tbt = self.get_integrals()
        h = numpy.zeros(shape=[n_orbitals] * 2)
        g = numpy.zeros(shape=[n_orbitals] * 2)
        for p in range(n_orbitals):
            h[p, p] += 2 * obt[p, p]
            for q in range(n_orbitals):
                h[p, q] += + tbt[p, p, q, q]
                if p != q:
                    g[p, q] += 2 * tbt[p, q, q, p] - tbt[p, q, p, q]

        H = c
        for p in range(n_orbitals):
            for q in range(n_orbitals):
                up = p
                uq = q
                H += h[p, q] * Sm(up) * Sp(uq) + g[p, q] * Sm(up) * Sp(up) * Sm(uq) * Sp(uq)

        return H

    def make_molecular_hamiltonian(self):
        if self.active_space:
            return self.molecule.get_molecular_hamiltonian(occupied_indices=self.active_space.frozen_reference_orbitals,
                                                           active_indices=self.active_space.active_orbitals)
        else:
            return self.molecule.get_molecular_hamiltonian()

    def get_integrals(self, two_body_ordering="openfermion"):
        """
        Returns
        -------
        Tuple with:
        constant part (nuclear_repulsion + possible integrated parts from active-spaces)
        one_body_integrals
        two_body_integrals

        """
        if self.active_space is not None and len(self.active_space.frozen_reference_orbitals) > 0:
            c, h1, h2 = self.molecule.get_active_space_integrals(active_indices=self.active_space.active_orbitals,
                                                                occupied_indices=self.active_space.frozen_reference_orbitals)
        else:
            c = 0.0
            h1 = self.molecule.one_body_integrals
            h2 = self.molecule.two_body_integrals
        c += self.molecule.nuclear_repulsion
        h2 = NBodyTensor(h2, ordering="openfermion")
        h2 = h2.reorder(to=two_body_ordering).elems

        return c, h1, h2

    def compute_one_body_integrals(self):
        """ convenience function """
        c, h1, h2 = self.get_integrals()
        return h1

    def compute_two_body_integrals(self, two_body_ordering="openfermion"):
        """ """
        c, h1, h2 = self.get_integrals(two_body_ordering=two_body_ordering)
        return h2

    def compute_constant_part(self):
        c, h1, h2 = self.get_integrals()
        return c

    def compute_ccsd_amplitudes(self) -> ClosedShellAmplitudes:
        """ """
        raise Exception("BaseClass Method")

    def prepare_reference(self, state=None, *args, **kwargs):
        """

        Returns
        -------
        A tequila circuit object which prepares the reference of this molecule in the chosen transformation
        """

        if state is None:
            state = [1 for i in range(self.n_electrons)]
            state += [0 for i in range(2 * self.n_orbitals - self.n_electrons)]
        reference_state = BitString.from_array(self.transformation.map_state(state=state))
        return prepare_product_state(reference_state)

    def prepare_hcb_reference(self, state=None, *args, **kwargs):
        """

        Returns
        -------
        A tequila circuit object which prepares the reference of this molecule in hardcore-boson representation
        (a pair function represented only by the spin-up orbitals)
        this is independent of the qubit encoding (except the up_then_down key) and can be transformed via
        U = self.transfomration.hcb_to_me
        so
        self.prepare_reference == self.prepare_hcb_reference + self.transformation.hcb_to_me()

        state can define a given product state (expected in full spin orbital notation up, down, up, down)
        """

        if state is None:
            state = [1 for i in range(self.n_electrons)]
            state += [0 for i in range(2 * self.n_orbitals - self.n_electrons)]
        reference_state = [0] * len(state)
        for i in range(self.n_orbitals):
            assert state[2 * i] == state[2 * i + 1]
            reference_state[self.transformation.up(i)] = state[2 * i]

        return prepare_product_state(BitString.from_array(reference_state))

    def prepare_hardcore_boson_reference(self):
        # todo: integrate with transformation
        return gates.X(target=[i for i in range(self.n_electrons // 2)])

    def hcb_to_me(self, U=None):
        """
        Transform a circuit in the hardcore-boson encoding (HCB)
        to the encoding of this molecule
        HCB is supposed to be encoded on the first n_orbitals qubits
        Parameters
        ----------
        U: HCB circuit (using the alpha qubits)
        Returns
        -------

        """
        if U is None:
            U = QCircuit()

        # consistency
        consistency = [x < self.n_orbitals for x in U.qubits]
        if not all(consistency):
            warnings.warn(
                "hcb_to_me: given circuit is not defined on the first {} qubits. Is this a HCB circuit?".format(
                    self.n_orbitals))

        # map to alpha qubits
        alpha_map = {k: self.transformation.up(k) for k in range(self.n_orbitals)}
        alpha_U = U.map_qubits(qubit_map=alpha_map)
        UX = self.transformation.hcb_to_me()
        if UX is None:
            raise TequilaException(
                "transformation={} has no hcb_to_me function implemented".format(self.transformation))
        return alpha_U + UX

    def get_pair_specific_indices(self,
                                  pair_info: str = None,
                                  include_singles: bool = True,
                                  general_excitations: bool = True) -> list:
        """
        Assuming a pair-specific model, create a pair-specific index list
        to be used in make_upccgsd_ansatz(indices = ... )
        Excite from a set of references (i) to any pair coming from (i),
        i.e. any (i,j)/(j,i). If general excitations are allowed, also
        allow excitations from pairs to appendant pairs and reference.

        Parameters
        ----------
        pair_info
            file or list including information about pair structure
            references single number, pair double
            example: as file: "0,1,11,11,00,10" (hand over file name)
                     in file, skip first row assuming some text with information
                     as list:['0','1`','11','11','00','10']
                     ~> two reference orbitals 0 and 1,
                     then two orbitals from pair 11, one from 00, one mixed 10
        include_singles
            include single excitations
        general_excitations
            allow general excitations
       Returns
        -------
            list of indices with pair-specific ansatz
        """

        if pair_info is None:
            raise TequilaException("Need to provide some pair information.")
        # If pair-information given on file, load (layout see above)
        if isinstance(pair_info, str):
            pairs = numpy.loadtxt(pair_info, dtype=str, delimiter=",", skiprows=1)
        elif isinstance(pair_info, list):
            pairs = pair_info
        elif not isinstance(pair_info, list):
            raise TequilaException("Pair information needs to be contained in a list or filename.")

        connect = [[]] * len(pairs)
        # determine "connectivity"
        generalized = 0
        for idx, p in enumerate(pairs):
            if len(p) == 1:
                connect[idx] = [i for i in range(len(pairs))
                                if ((len(pairs[i]) == 2) and (str(idx) in pairs[i]))]
            elif (len(p) == 2) and general_excitations:
                connect[idx] = [i for i in range(len(pairs))
                                if (((p[0] in pairs[i]) or (p[1] in pairs[i]) or str(i) in p)
                                    and not (i == idx))]
            elif len(p) > 2:
                raise TequilaException("Invalid reference of pair id.")

        # create generating indices from connectivity
        indices = []
        for i, to in enumerate(connect):
            for a in to:
                indices.append(((2 * i, 2 * a), (2 * i + 1, 2 * a + 1)))
                if include_singles:
                    indices.append(((2 * i, 2 * a)))
                    indices.append(((2 * i + 1, 2 * a + 1)))

        return indices

    def format_excitation_indices(self, idx):
        """
        Consistent formatting of excitation indices
        idx = [(p0,q0),(p1,q1),...,(pn,qn)]
        sorted as: p0<p1<pn and pi<qi
        :param idx: list of index tuples describing a single(!) fermionic excitation
        :return: tuple-list of index tuples
        """

        idx = [tuple(sorted(x)) for x in idx]
        idx = sorted(idx, key=lambda x: x[0])
        return tuple(idx)

    def make_upccgsd_indices(self, key, reference_orbitals=None, *args, **kwargs):

        if reference_orbitals is None:
            reference_orbitals = [i for i in range(self.n_electrons // 2)]
        indices = []
        # add doubles in hcb encoding
        if hasattr(key, "lower") and key.lower() == "ladder":
            # ladder structure of the pair excitations
            # ensures local connectivity
            indices = [[(n, n + 1)] for n in range(self.n_orbitals - 1)]
        elif hasattr(key, "lower") and "g" not in key:
            indices = [[(n, m)] for n in reference_orbitals for m in range(self.n_orbitals) if
                       n < m and m not in reference_orbitals]
        elif hasattr(key, "lower") and "g" in key:
            indices = [[(n, m)] for n in range(self.n_orbitals) for m in range(self.n_orbitals) if n < m]
        else:
            raise TequilaException("Unknown recipe: {}".format(key))

        indices = [self.format_excitation_indices(idx) for idx in indices]

        return indices

    def make_hardcore_boson_upccgd_layer(self,
                                         indices: list = "UpCCGD",
                                         label: str = None,
                                         assume_real: bool = True,
                                         *args, **kwargs):

        if hasattr(indices, "lower"):
            indices = self.make_upccgsd_indices(key=indices.lower())

        UD = QCircuit()
        for idx in indices:
            UD += self.make_hardcore_boson_excitation_gate(indices=idx, angle=(idx, "D", label),
                                                           assume_real=assume_real)

        return UD

    def make_upccgsd_ansatz(self,
                            include_reference: bool = True,
                            name: str = "UpCCGSD",
                            label: str = None,
                            order: int = None,
                            assume_real: bool = True,
                            use_hcb: bool = None,
                            spin_adapt_singles: bool = True,
                            neglect_z = False,
                            *args, **kwargs):
        """
        UpGCCSD Ansatz similar as described by Lee et. al.

        Parameters
        ----------
        include_singles
            include singles excitations. Is overwritten if indices are a string (i.e. indices=UpCCGSD will always include singles, UpCCGD will not)
        include_reference
            include the HF reference state as initial state
        indices
            pass custom defined set of indices from which the ansatz will be created
            List of tuples of tuples spin-indices e.g. [((2*p,2*q),(2*p+1,2*q+1)), ...]
        label
            An additional label that is set with the variables
            default is None and no label will be set: variables names will be
            (x, (p,q)) for x in range(order)
            with a label the variables will be named
            (label, (x, (p,q)))
        order
            Order of the ansatz (default is 1)
            determines how often the ordering gets repeated
            parameters of repeating layers are independent
        assume_real
            assume a real wavefunction (that is always the case if the reference state is real)
            reduces potential gradient costs from 4 to 2
        Returns
        -------
            UpGCCSD ansatz
        """

        name = name.upper()

        if ("A" in name) and neglect_z is None:
            neglect_z = True
        else:
            neglect_z = False

        if order is None:
            try:
                if "-" in name:
                    order = int(name.split("-")[0])
                else:
                    order = 1
            except:
                order = 1

        indices = self.make_upccgsd_indices(key=name)

        # check if the used qubit encoding has a hcb transformation
        have_hcb_trafo = self.transformation.hcb_to_me() is not None

        # consistency checks for optimization
        if have_hcb_trafo and use_hcb is None:
            use_hcb = True
        if "HCB" in name:
            use_hcb = True
        if use_hcb and not have_hcb_trafo and "HCB" not in name:
            raise TequilaException(
                "use_hcb={} but transformation={} has no \'hcb_to_me\' function. Try transformation=\'ReorderedJordanWigner\'".format(
                    use_hcb, self.transformation))
        if "S" in name and "HCB" in name:
            if "HCB" in name and "S" in name:
                raise Exception(
                    "name={}, Singles can't be realized without mapping back to the standard encoding leave S or HCB out of the name".format(
                        name))

        # first layer
        if not use_hcb:
            U = QCircuit()
            if include_reference:
                U = self.prepare_reference()
            U += self.make_upccgsd_layer(include_singles="S" in name, indices=indices, assume_real=assume_real,
                                         label=(label, 0), spin_adapt_singles=spin_adapt_singles, *args, **kwargs)
        else:
            U = QCircuit()
            if include_reference:
                U = self.prepare_hardcore_boson_reference()
            U += self.make_hardcore_boson_upccgd_layer(indices=indices, assume_real=assume_real, label=(label, 0),
                                                       *args, **kwargs)
            if "HCB" not in name:
                U = self.hcb_to_me(U=U)

            if "S" in name:
                self.make_upccgsd_singles(indices=indices, assume_real=assume_real, label=(label, 0),
                                          spin_adapt_singles=spin_adapt_singles, neglect_z=neglect_z, *args, **kwargs)

        for k in range(1, order):
            U += self.make_upccgsd_layer(include_singles="S" in name, indices=indices, label=(label, k),
                                         spin_adapt_singles=spin_adapt_singles, neglect_z=neglect_z)

        return U

    def make_upccgsd_layer(self, indices, include_singles=True, include_doubles=True, assume_real=True, label=None,
                           spin_adapt_singles: bool = True, angle_transform=None, mix_sd=False, neglect_z=False, *args, **kwargs):
        U = QCircuit()
        for idx in indices:
            assert len(idx) == 1
            idx = idx[0]
            angle = (tuple([idx]), "D", label)
            if include_doubles:
                if  "jordanwigner" in self.transformation.name.lower():
                    # we can optimize with qubit excitations for the JW representation
                    target=[self.transformation.up(idx[0]), self.transformation.up(idx[1]), self.transformation.down(idx[0]), self.transformation.down(idx[1])]
                    U += gates.QubitExcitation(angle=angle, target=target)
                else:
                    U += self.make_excitation_gate(angle=angle,
                                                   indices=((2 * idx[0], 2 * idx[1]), (2 * idx[0] + 1, 2 * idx[1] + 1)),
                                                   assume_real=assume_real)
            if include_singles and mix_sd:
                U += self.make_upccgsd_singles(indices=[idx], assume_real=assume_real, label=label,
                                               spin_adapt_singles=spin_adapt_singles, angle_transform=angle_transform, neglect_z=neglect_z)

        if include_singles and not mix_sd:
            U += self.make_upccgsd_singles(indices=indices, assume_real=assume_real, label=label,
                                           spin_adapt_singles=spin_adapt_singles, angle_transform=angle_transform, neglect_z=neglect_z)
        return U

    def make_upccgsd_singles(self, indices="UpCCGSD", spin_adapt_singles=True, label=None, angle_transform=None,
                             assume_real=True, neglect_z=False):
        if neglect_z and "jordanwigner" not in self.transformation.name.lower():
            raise TequilaException("neglegt-z approximation in UpCCGSD singles needs the (Reversed)JordanWigner representation")
        if hasattr(indices, "lower"):
            indices = self.make_upccgsd_indices(key=indices)

        U = QCircuit()
        for idx in indices:
            assert len(idx) == 1
            idx = idx[0]
            if spin_adapt_singles:
                angle = (idx, "S", label)
                if angle_transform is not None:
                    angle = angle_transform(angle)
                if neglect_z:
                    targeta=[self.transformation.up(idx[0]), self.transformation.up(idx[1])]
                    targetb=[self.transformation.down(idx[0]), self.transformation.down(idx[1])]
                    U += gates.QubitExcitation(angle=angle, target=targeta, assume_real=assume_real)
                    U += gates.QubitExcitation(angle=angle, target=targetb, assume_real=assume_real)
                else:
                    U += self.make_excitation_gate(angle=angle, indices=[(2 * idx[0], 2 * idx[1])], assume_real=assume_real)
                    U += self.make_excitation_gate(angle=angle, indices=[(2 * idx[0] + 1, 2 * idx[1] + 1)],
                                               assume_real=assume_real)
            else:
                angle1 = (idx, "SU", label)
                angle2 = (idx, "SD", label)
                if angle_transform is not None:
                    angle1 = angle_transform(angle1)
                    angle2 = angle_transform(angle2)
                if neglect_z:
                    targeta=[self.transformation.up(idx[0]), self.transformation.up(idx[1])]
                    targetb=[self.transformation.down(idx[0]), self.transformation.down(idx[1])]
                    U += gates.QubitExcitation(angle=angle1, target=targeta, assume_real=assume_real)
                    U += gates.QubitExcitation(angle=angle2, target=targetb, assume_real=assume_real)
                else:
                    U += self.make_excitation_gate(angle=angle1, indices=[(2 * idx[0], 2 * idx[1])],
                                               assume_real=assume_real)
                    U += self.make_excitation_gate(angle=angle2, indices=[(2 * idx[0] + 1, 2 * idx[1] + 1)],
                                               assume_real=assume_real)

        return U

    def make_uccsd_ansatz(self, trotter_steps: int,
                          initial_amplitudes: typing.Union[str, Amplitudes, ClosedShellAmplitudes] = "mp2",
                          include_reference_ansatz=True,
                          parametrized=True,
                          threshold=1.e-8,
                          *args, **kwargs) -> QCircuit:
        """

        Parameters
        ----------
        initial_amplitudes :
            initial amplitudes given as ManyBodyAmplitudes structure or as string
            where 'mp2', 'cc2' or 'ccsd' are possible initializations
        include_reference_ansatz :
            Also do the reference ansatz (prepare closed-shell Hartree-Fock) (Default value = True)
        parametrized :
            Initialize with variables, otherwise with static numbers (Default value = True)
        trotter_steps: int :

        initial_amplitudes: typing.Union[str :

        Amplitudes :

        ClosedShellAmplitudes] :
             (Default value = "mp2")

        Returns
        -------
        type
            Parametrized QCircuit

        """

        if self.n_electrons % 2 != 0:
            raise TequilaException("make_uccsd_ansatz currently only for closed shell systems")

        nocc = self.n_electrons // 2
        nvirt = self.n_orbitals // 2 - nocc

        Uref = QCircuit()
        if include_reference_ansatz:
            Uref = self.prepare_reference()

        amplitudes = initial_amplitudes
        if hasattr(initial_amplitudes, "lower"):
            if initial_amplitudes.lower() == "mp2":
                amplitudes = self.compute_mp2_amplitudes()
            elif initial_amplitudes.lower() == "ccsd":
                amplitudes = self.compute_ccsd_amplitudes()
            else:
                try:
                    amplitudes = self.compute_amplitudes(method=initial_amplitudes.lower())
                except Exception as exc:
                    raise TequilaException(
                        "{}\nDon't know how to initialize \'{}\' amplitudes".format(exc, initial_amplitudes))

        if amplitudes is None:
            amplitudes = ClosedShellAmplitudes(
                tIjAb=numpy.zeros(shape=[nocc, nocc, nvirt, nvirt]),
                tIA=numpy.zeros(shape=[nocc, nvirt]))

        closed_shell = isinstance(amplitudes, ClosedShellAmplitudes)
        indices = []
        variables = []

        if not isinstance(amplitudes, dict):
            amplitudes = amplitudes.make_parameter_dictionary(threshold=threshold)
            amplitudes = dict(sorted(amplitudes.items(), key=lambda x: numpy.fabs(x[1]), reverse=True))

        for key, t in amplitudes.items():
            assert (len(key) % 2 == 0)
            if not numpy.isclose(t, 0.0, atol=threshold):

                if closed_shell:
                    spin_indices = []
                    if len(key) == 2:
                        spin_indices = [[2 * key[0], 2 * key[1]], [2 * key[0] + 1, 2 * key[1] + 1]]
                        partner = None
                    else:
                        spin_indices.append([2 * key[0] + 1, 2 * key[1] + 1, 2 * key[2], 2 * key[3]])
                        #spin_indices.append([2 * key[0], 2 * key[1], 2 * key[2] + 1, 2 * key[3] + 1])
                        if key[0] != key[2] and key[1] != key[3]:
                            spin_indices.append([2 * key[0], 2 * key[1], 2 * key[2], 2 * key[3]])
                            #spin_indices.append([2 * key[0] + 1, 2 * key[1] + 1, 2 * key[2] + 1, 2 * key[3] + 1])
                        partner = tuple([key[2], key[1], key[0], key[3]])  # taibj -> tbiaj
                    for idx in spin_indices:
                        idx = [(idx[2 * i], idx[2 * i + 1]) for i in range(len(idx) // 2)]
                        indices.append(idx)

                    if parametrized:
                        variables.append(2.0*Variable(name=key))  # abab
                        #variables.append(Variable(name=key))  # baba
                        if partner is not None and key[0] != key[1] and key[2] != key[3]:
                            variables.append(2.0*(Variable(name=key) - Variable(partner))) # aaaa
                            #variables.append(Variable(name=key) - Variable(partner))  # bbbb
                    else:
                        variables.append(2.0*t)
                        #variables.append(t)
                        if partner is not None and key[0] != key[1] and key[2] != key[3]:
                            variables.append(2.0*(t - amplitudes[partner]))
                            #variables.append(t - amplitudes[partner])
                else:
                    indices.append(spin_indices)
                    if parametrized:
                        variables.append(Variable(name=key))
                    else:
                        variables.append(t)
        UCCSD = QCircuit()
        factor = 1.0 / trotter_steps
        for step in range(trotter_steps):
            for i, idx in enumerate(indices):
                UCCSD += self.make_excitation_gate(indices=idx, angle=factor * variables[i])

        return Uref + UCCSD

    def compute_amplitudes(self, method: str, *args, **kwargs):
        """
        Compute closed-shell CC amplitudes

        Parameters
        ----------
        method :
            coupled-cluster methods like cc2, ccsd, cc3, ccsd(t)
            Success might depend on backend
            got an extra function for MP2
        *args :

        **kwargs :


        Returns
        -------

        """
        raise TequilaException("compute amplitudes: Needs to be overwritten by backend")

    def compute_mp2_amplitudes(self) -> ClosedShellAmplitudes:
        """

        Compute closed-shell mp2 amplitudes

        .. math::
            t(a,i,b,j) = 0.25 * g(a,i,b,j)/(e(i) + e(j) -a(i) - b(j) )

        :return:

        Parameters
        ----------

        Returns
        -------

        """
        assert self.parameters.closed_shell
        g = self.molecule.two_body_integrals
        fij = self.molecule.orbital_energies
        nocc = self.molecule.n_electrons // 2  # this is never the active space
        ei = fij[:nocc]
        ai = fij[nocc:]
        abgij = g[nocc:, nocc:, :nocc, :nocc]
        amplitudes = abgij * 1.0 / (
                ei.reshape(1, 1, -1, 1) + ei.reshape(1, 1, 1, -1) - ai.reshape(-1, 1, 1, 1) - ai.reshape(1, -1, 1, 1))
        E = 2.0 * numpy.einsum('abij,abij->', amplitudes, abgij) - numpy.einsum('abji,abij', amplitudes, abgij,
                                                                                optimize='greedy')

        self.molecule.mp2_energy = E + self.molecule.hf_energy
        return ClosedShellAmplitudes(tIjAb=numpy.einsum('abij -> ijab', amplitudes, optimize='greedy'))

    def compute_cis_amplitudes(self):
        """
        Compute the CIS amplitudes of the molecule
        """

        @dataclass
        class ResultCIS:
            """ """
            omegas: typing.List[numbers.Real]  # excitation energies [omega0, ...]
            amplitudes: typing.List[ClosedShellAmplitudes]  # corresponding amplitudes [x_{ai}_0, ...]

            def __getitem__(self, item):
                return (self.omegas[item], self.amplitudes[item])

            def __len__(self):
                return len(self.omegas)

        g = self.molecule.two_body_integrals
        fij = self.molecule.orbital_energies

        nocc = self.n_alpha_electrons
        nvirt = self.n_orbitals - nocc

        pairs = []
        for i in range(nocc):
            for a in range(nocc, nocc + nvirt):
                pairs.append((a, i))
        M = numpy.ndarray(shape=[len(pairs), len(pairs)])

        for xx, x in enumerate(pairs):
            eia = fij[x[0]] - fij[x[1]]
            a, i = x
            for yy, y in enumerate(pairs):
                b, j = y
                delta = float(y == x)
                gpart = 2.0 * g[a, i, b, j] - g[a, i, j, b]
                M[xx, yy] = eia * delta + gpart

        omega, xvecs = numpy.linalg.eigh(M)

        # convert amplitudes to ndarray sorted by excitation energy
        nex = len(omega)
        amplitudes = []
        for ex in range(nex):
            t = numpy.ndarray(shape=[nvirt, nocc])
            exvec = xvecs[ex]
            for xx, x in enumerate(pairs):
                a, i = x
                t[a - nocc, i] = exvec[xx]
            amplitudes.append(ClosedShellAmplitudes(tIA=t))

        return ResultCIS(omegas=list(omega), amplitudes=amplitudes)

    @property
    def rdm1(self):
        """ """
        if self._rdm1 is not None:
            return self._rdm1
        else:
            print("1-RDM has not been computed. Return None for 1-RDM.")
            return None

    @property
    def rdm2(self):
        """ """
        if self._rdm2 is not None:
            return self._rdm2
        else:
            print("2-RDM has not been computed. Return None for 2-RDM.")
            return None

    def compute_rdms(self, U: QCircuit = None, variables: Variables = None, spin_free: bool = True,
                     get_rdm1: bool = True, get_rdm2: bool = True):
        """
        Computes the one- and two-particle reduced density matrices (rdm1 and rdm2) given
        a unitary U. This method uses the standard ordering in physics as denoted below.
        Note, that the representation of the density matrices depends on the qubit transformation
        used. The Jordan-Wigner encoding corresponds to 'classical' second quantized density
        matrices in the occupation picture.

        We only consider real orbitals and thus real-valued RDMs.
        The matrices are set as private members _rdm1, _rdm2 and can be accessed via the properties rdm1, rdm2.

        .. math :
            \\text{rdm1: } \\gamma^p_q = \\langle \\psi | a^p a_q | \\psi \\rangle
                                     = \\langle U 0 | a^p a_q | U 0 \\rangle
            \\text{rdm2: } \\gamma^{pq}_{rs} = \\langle \\psi | a^p a^q a_s a_r | \\psi \\rangle
                                             = \\langle U 0 | a^p a^q a_s a_r | U 0 \\rangle

        Parameters
        ----------
        U :
            Quantum Circuit to achieve the desired state \\psi = U |0\\rangle, non-optional
        variables :
            If U is parametrized, then need to hand over a set of fixed variables
        spin_free :
            Set whether matrices should be spin-free (summation over spin) or defined by spin-orbitals
        get_rdm1, get_rdm2 :
            Set whether either one or both rdm1, rdm2 should be computed. If both are needed at some point,
            it is recommended to compute them at once.

        Returns
        -------
        """
        # Check whether unitary circuit is not 0
        if U is None:
            raise TequilaException('Need to specify a Quantum Circuit.')

        # Check whether transformation is BKSF.
        # Issue here: when a single operator acts only on a subset of qubits, BKSF might not yield the correct
        # transformation, because it computes the number of qubits incorrectly in this case.
        # A hotfix such as for symmetry_conserving_bravyi_kitaev would require deeper changes, thus omitted for now
        if type(self.transformation).__name__ == "BravyiKitaevFast":
            raise TequilaException(
                "The Bravyi-Kitaev-Superfast transformation does not support general FermionOperators yet.")

        # Set up number of spin-orbitals and molecular orbitals respectively
        n_SOs = 2 * self.n_orbitals
        n_MOs = self.n_orbitals

        # Check whether unitary circuit is not 0
        if U is None:
            raise TequilaException('Need to specify a Quantum Circuit.')

        def _get_of_op(operator_tuple):
            """ Returns operator given by a operator tuple as OpenFermion - Fermion operator """
            op = openfermion.FermionOperator(operator_tuple)
            return op

        def _get_qop_hermitian(of_operator) -> QubitHamiltonian:
            """ Returns Hermitian part of Fermion operator as QubitHamiltonian """
            qop = QubitHamiltonian(self.transformation(of_operator))
            real, imag = qop.split(hermitian=True)
            if real:
                return real
            elif not real:
                raise TequilaException(
                    "Qubit Hamiltonian does not have a Hermitian part. Operator ={}".format(of_operator))

        def _build_1bdy_operators_spinful() -> list:
            """ Returns spinful one-body operators as a symmetry-reduced list of QubitHamiltonians """
            # Exploit symmetry pq = qp
            ops = []
            for p in range(n_SOs):
                for q in range(p + 1):
                    op_tuple = ((p, 1), (q, 0))
                    op = _get_of_op(op_tuple)
                    ops += [op]

            return ops

        def _build_2bdy_operators_spinful() -> list:
            """ Returns spinful two-body operators as a symmetry-reduced list of QubitHamiltonians """
            # Exploit symmetries pqrs = -pqsr = -qprs = qpsr
            #                and      =  rspq
            ops = []
            for p in range(n_SOs):
                for q in range(p):
                    for r in range(n_SOs):
                        for s in range(r):
                            if p * n_SOs + q >= r * n_SOs + s:
                                op_tuple = ((p, 1), (q, 1), (s, 0), (r, 0))
                                op = _get_of_op(op_tuple)
                                ops += [op]

            return ops

        def _build_1bdy_operators_spinfree() -> list:
            """ Returns spinfree one-body operators as a symmetry-reduced list of QubitHamiltonians """
            # Exploit symmetry pq = qp (not changed by spin-summation)
            ops = []
            for p in range(n_MOs):
                for q in range(p + 1):
                    # Spin aa
                    op_tuple = ((2 * p, 1), (2 * q, 0))
                    op = _get_of_op(op_tuple)
                    # Spin bb
                    op_tuple = ((2 * p + 1, 1), (2 * q + 1, 0))
                    op += _get_of_op(op_tuple)
                    ops += [op]

            return ops

        def _build_2bdy_operators_spinfree() -> list:
            """ Returns spinfree two-body operators as a symmetry-reduced list of QubitHamiltonians """
            # Exploit symmetries pqrs = qpsr (due to spin summation, '-pqsr = -qprs' drops out)
            #                and      = rspq
            ops = []
            for p, q, r, s in product(range(n_MOs), repeat=4):
                if p * n_MOs + q >= r * n_MOs + s and (p >= q or r >= s):
                    # Spin aaaa
                    op_tuple = ((2 * p, 1), (2 * q, 1), (2 * s, 0), (2 * r, 0)) if (p != q and r != s) else '0.0 []'
                    op = _get_of_op(op_tuple)
                    # Spin abab
                    op_tuple = ((2 * p, 1), (2 * q + 1, 1), (2 * s + 1, 0), (2 * r, 0)) if (
                            2 * p != 2 * q + 1 and 2 * r != 2 * s + 1) else '0.0 []'
                    op += _get_of_op(op_tuple)
                    # Spin baba
                    op_tuple = ((2 * p + 1, 1), (2 * q, 1), (2 * s, 0), (2 * r + 1, 0)) if (
                            2 * p + 1 != 2 * q and 2 * r + 1 != 2 * s) else '0.0 []'
                    op += _get_of_op(op_tuple)
                    # Spin bbbb
                    op_tuple = ((2 * p + 1, 1), (2 * q + 1, 1), (2 * s + 1, 0), (2 * r + 1, 0)) if (
                            p != q and r != s) else '0.0 []'
                    op += _get_of_op(op_tuple)

                    ops += [op]

            return ops

        def _assemble_rdm1(evals) -> numpy.ndarray:
            """
            Returns spin-ful or spin-free one-particle RDM built by symmetry conditions
            Same symmetry with or without spin, so we can use the same function
            """
            N = n_MOs if spin_free else n_SOs
            rdm1 = numpy.zeros([N, N])
            ctr: int = 0
            for p in range(N):
                for q in range(p + 1):
                    rdm1[p, q] = evals[ctr]
                    # Symmetry pq = qp
                    rdm1[q, p] = rdm1[p, q]
                    ctr += 1

            return rdm1

        def _assemble_rdm2_spinful(evals) -> numpy.ndarray:
            """ Returns spin-ful two-particle RDM built by symmetry conditions """
            ctr: int = 0
            rdm2 = numpy.zeros([n_SOs, n_SOs, n_SOs, n_SOs])
            for p in range(n_SOs):
                for q in range(p):
                    for r in range(n_SOs):
                        for s in range(r):
                            if p * n_SOs + q >= r * n_SOs + s:
                                rdm2[p, q, r, s] = evals[ctr]
                                # Symmetry pqrs = rspq
                                rdm2[r, s, p, q] = rdm2[p, q, r, s]
                                ctr += 1

            # Further permutational symmetries due to anticommutation relations
            for p in range(n_SOs):
                for q in range(p):
                    for r in range(n_SOs):
                        for s in range(r):
                            rdm2[p, q, s, r] = -1 * rdm2[p, q, r, s]  # pqrs = -pqsr
                            rdm2[q, p, r, s] = -1 * rdm2[p, q, r, s]  # pqrs = -qprs
                            rdm2[q, p, s, r] = rdm2[p, q, r, s]  # pqrs =  qpsr

            return rdm2

        def _assemble_rdm2_spinfree(evals) -> numpy.ndarray:
            """ Returns spin-free two-particle RDM built by symmetry conditions """
            ctr: int = 0
            rdm2 = numpy.zeros([n_MOs, n_MOs, n_MOs, n_MOs])
            for p, q, r, s in product(range(n_MOs), repeat=4):
                if p * n_MOs + q >= r * n_MOs + s and (p >= q or r >= s):
                    rdm2[p, q, r, s] = evals[ctr]
                    # Symmetry pqrs = rspq
                    rdm2[r, s, p, q] = rdm2[p, q, r, s]
                    ctr += 1

            # Further permutational symmetry: pqrs = qpsr
            for p, q, r, s in product(range(n_MOs), repeat=4):
                if p >= q or r >= s:
                    rdm2[q, p, s, r] = rdm2[p, q, r, s]

            return rdm2

        # Build operator lists
        qops = []
        if spin_free:
            qops += _build_1bdy_operators_spinfree() if get_rdm1 else []
            qops += _build_2bdy_operators_spinfree() if get_rdm2 else []
        else:
            qops += _build_1bdy_operators_spinful() if get_rdm1 else []
            qops += _build_2bdy_operators_spinful() if get_rdm2 else []

        # Transform operator lists to QubitHamiltonians
        qops = [_get_qop_hermitian(op) for op in qops]
        # Compute expected values
        evals = simulate(ExpectationValue(H=qops, U=U, shape=[len(qops)]), variables=variables)

        # Assemble density matrices
        # If self._rdm1, self._rdm2 exist, reset them if they are of the other spin-type
        def _reset_rdm(rdm):
            if rdm is not None:
                if spin_free and rdm.shape[0] != n_MOs:
                    return None
                if not spin_free and rdm.shape[0] != n_SOs:
                    return None
            return rdm

        self._rdm1 = _reset_rdm(self._rdm1)
        self._rdm2 = _reset_rdm(self._rdm2)
        # Split expectation values in 1- and 2-particle expectation values
        if get_rdm1:
            len_1 = n_MOs * (n_MOs + 1) // 2 if spin_free else n_SOs * (n_SOs + 1) // 2
        else:
            len_1 = 0
        evals_1, evals_2 = evals[:len_1], evals[len_1:]
        # Build matrices using the expectation values
        self._rdm1 = _assemble_rdm1(evals_1) if get_rdm1 else self._rdm1
        if spin_free:
            self._rdm2 = _assemble_rdm2_spinfree(evals_2) if get_rdm2 else self._rdm2
        else:
            self._rdm2 = _assemble_rdm2_spinful(evals_2) if get_rdm2 else self._rdm2

    def rdm_spinsum(self, sum_rdm1: bool = True, sum_rdm2: bool = True) -> tuple:
        """
        Given the spin-ful 1- and 2-particle reduced density matrices, compute the spin-free RDMs by spin summation.

        Parameters
        ----------
            sum_rdm1, sum_rdm2 :
               If set to true, perform spin summation on rdm1, rdm2

        Returns
        -------
            rdm1_spinsum, rdm2_spinsum :
                The desired spin-free matrices
        """
        n_MOs = self.n_orbitals
        rdm1_spinsum = None
        rdm2_spinsum = None

        # Spin summation on rdm1
        if sum_rdm1:
            # Check whether spin-rdm2 exists
            if self._rdm1 is None:
                raise TequilaException("The spin-RDM for the 1-RDM does not exist!")
            # Check whether existing rdm1 is in spin-orbital basis
            if self._rdm1.shape[0] != 2 * n_MOs:
                raise TequilaException("The existing RDM needs to be in spin-orbital basis, it is already spin-free!")
            # Do summation
            rdm1_spinsum = numpy.zeros([n_MOs, n_MOs])
            for p in range(n_MOs):
                for q in range(p + 1):
                    rdm1_spinsum[p, q] += self._rdm1[2 * p, 2 * q]
                    rdm1_spinsum[p, q] += self._rdm1[2 * p + 1, 2 * q + 1]
            for p in range(n_MOs):
                for q in range(p):
                    rdm1_spinsum[q, p] = rdm1_spinsum[p, q]

        # Spin summation on rdm2
        if sum_rdm2:
            # Check whether spin-rdm2 exists
            if self._rdm2 is None:
                raise TequilaException("The spin-RDM for the 2-RDM does not exist!")
            # Check whether existing rdm2 is in spin-orbital basis
            if self._rdm2.shape[0] != 2 * n_MOs:
                raise TequilaException("The existing RDM needs to be in spin-orbital basis, it is already spin-free!")
            # Do summation
            rdm2_spinsum = numpy.zeros([n_MOs, n_MOs, n_MOs, n_MOs])
            for p, q, r, s in product(range(n_MOs), repeat=4):
                rdm2_spinsum[p, q, r, s] += self._rdm2[2 * p, 2 * q, 2 * r, 2 * s]
                rdm2_spinsum[p, q, r, s] += self._rdm2[2 * p + 1, 2 * q, 2 * r + 1, 2 * s]
                rdm2_spinsum[p, q, r, s] += self._rdm2[2 * p, 2 * q + 1, 2 * r, 2 * s + 1]
                rdm2_spinsum[p, q, r, s] += self._rdm2[2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1]

        return rdm1_spinsum, rdm2_spinsum

    def __str__(self) -> str:
        result = str(type(self)) + "\n"
        result += "Qubit Encoding\n"
        result += str(self.transformation) + "\n\n"
        result += "Parameters\n"
        for k, v in self.parameters.__dict__.items():
            result += "{key:15} : {value:15} \n".format(key=str(k), value=str(v))
        result += "\n"
        return result
