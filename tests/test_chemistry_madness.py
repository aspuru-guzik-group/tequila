import pytest
import numpy
import os
import tequila as tq

from tequila.quantumchemistry import INSTALLED_QCHEMISTRY_BACKENDS
has_pyscf = "pyscf" in INSTALLED_QCHEMISTRY_BACKENDS

root = os.environ.get("MAD_ROOT_DIR")
executable = tq.quantumchemistry.madness_interface.QuantumChemistryMadness.find_executable(root)
print("root = ", root)
print("executable = ", executable)


def test_executable():
    if root is not None and executable is None:
        raise Exception("Found no pno_integrals executable but found MAD_ROOT_DIR={}\n"
                        "Seems like you wanted that tested".format(root))


@pytest.mark.skipif(not (os.path.isfile('he_gtensor.npy') and os.path.isfile('he_htensor.npy')),
                    reason="data not there")
def test_madness_he_data():
    # relies that he_xtensor.npy are present (x=g,h)
    geomstring = "He 0.0 0.0 0.0"
    molecule = tq.Molecule(name="he", geometry=geomstring)
    H = molecule.make_hamiltonian()
    UHF = molecule.prepare_reference()
    EHF = tq.simulate(tq.ExpectationValue(H=H, U=UHF))
    assert (numpy.isclose(-2.861522e+00, EHF, atol=1.e-5))
    U = molecule.make_upccgsd_ansatz()
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(method="bfgs", objective=E, initial_values=0.0, silent=True)
    print(result.energy)
    assert (numpy.isclose(-2.87761809, result.energy, atol=1.e-5))


@pytest.mark.skipif(executable is None, reason="madness was not found")
def test_madness_full_he():
    # relies on madness being compiled and MAD_ROOT_DIR exported
    # or pno_integrals in the path
    geomstring = "He 0.0 0.0 0.0"
    molecule = tq.Molecule(geometry=geomstring, n_pno=1)
    H = molecule.make_hamiltonian()
    UHF = molecule.prepare_reference()
    EHF = tq.simulate(tq.ExpectationValue(H=H, U=UHF))
    assert (numpy.isclose(-2.861522e+00, EHF, atol=1.e-5))
    U = molecule.make_upccgsd_ansatz()
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(method="bfgs", objective=E, initial_values=0.0, silent=True)
    assert (numpy.isclose(-2.87761809, result.energy, atol=1.e-5))


@pytest.mark.skipif(executable is None, reason="madness was not found")
def test_madness_full_li_plus():
    # relies on madness being compiled and MAD_ROOT_DIR exported
    # or pno_integrals in the path
    geomstring = "Li 0.0 0.0 0.0"
    molecule = tq.Molecule(name="li+", geometry=geomstring, n_pno=1, charge=1)
    H = molecule.make_hamiltonian()
    UHF = molecule.prepare_reference()
    EHF = tq.simulate(tq.ExpectationValue(H=H, U=UHF))
    assert (numpy.isclose(-7.236247e+00, EHF, atol=1.e-5))
    U = molecule.make_upccgsd_ansatz()
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(method="bfgs", objective=E, initial_values=0.0, silent=True)
    assert (numpy.isclose(-7.251177798, result.energy, atol=1.e-5))


@pytest.mark.skipif(executable is None, reason="madness was not found")
def test_madness_full_be():
    # relies on madness being compiled and MAD_ROOT_DIR exported
    # or pno_integrals in the path
    geomstring = "Be 0.0 0.0 0.0"
    molecule = tq.Molecule(name="be", geometry=geomstring, n_pno=3, frozen_core=True)
    H = molecule.make_hamiltonian()
    UHF = molecule.prepare_reference()
    EHF = tq.simulate(tq.ExpectationValue(H=H, U=UHF))
    assert (numpy.isclose(-14.57269300, EHF, atol=1.e-5))
    U = molecule.make_upccgsd_ansatz()
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(method="bfgs", objective=E, initial_values=0.0, silent=True)
    assert (numpy.isclose(-14.614662051580321, result.energy, atol=1.e-3))


@pytest.mark.parametrize("trafo", ["BravyiKitaev", "JordanWigner", "BravyiKitaevTree", "ReorderedJordanWigner",
                                   "ReorderedBravyiKitaev"])
@pytest.mark.skipif(executable is None and not os.path.isfile('balanced_be_gtensor.npy'),
                    reason="madness was not found")
def test_madness_upccgsd(trafo):
    n_pno = 2
    if os.path.isfile('balanced_be_gtensor.npy'):
        n_pno = None
    mol = tq.Molecule(name="balanced_be", geometry="Be 0.0 0.0 0.0", n_pno=n_pno, pno={"diagonal": True, "maxrank": 1},
                      transformation=trafo)

    H = mol.make_hardcore_boson_hamiltonian()
    oigawert=numpy.linalg.eigvalsh(H.to_matrix())[0]
    U = mol.make_upccgsd_ansatz(name="HCB-UpCCGD", direct_compiling=True)
    E = tq.ExpectationValue(H=H, U=U)
    assert (len(E.extract_variables()) == 6)
    result = tq.minimize(E)
    assert numpy.isclose(result.energy, oigawert, atol=1.e-3)

    U = mol.make_upccgsd_ansatz(name="HCB-PNO-UpCCD")
    E = tq.ExpectationValue(H=H, U=U)
    assert (len(E.extract_variables()) == 2)
    result = tq.minimize(E)
    assert numpy.isclose(result.energy, oigawert, atol=1.e-3)

    H = mol.make_hamiltonian()
    oigawert2=numpy.linalg.eigvalsh(H.to_matrix())[0]
    U = mol.make_upccgsd_ansatz(name="SPA-D")
    E = tq.ExpectationValue(H=H, U=U)
    assert (len(E.extract_variables()) == 2)
    variables = result.variables
    if "bravyi" in trafo.lower():
        # signs of angles change in BK compared to JW-like HCB
        variables = {k: -v for k, v in variables.items()}
    energy = tq.simulate(E, variables)
    result = tq.minimize(E)
    assert numpy.isclose(result.energy, oigawert, atol=1.e-3)
    assert numpy.isclose(energy, oigawert, atol=1.e-3)
    e3 = mol.compute_energy(method="SPA")
    assert numpy.isclose(result.energy, e3)
    e3 = mol.compute_energy(method="HCB-SPA")
    assert numpy.isclose(result.energy, e3)
    e3 = mol.compute_energy(method="SPA-UpCCD")
    assert numpy.isclose(result.energy, e3)

    U = mol.make_upccgsd_ansatz(name="SPA-UpCCSD")
    E = tq.ExpectationValue(H=H, U=U)
    assert (len(E.extract_variables()) == 4)
    result = tq.minimize(E)
    assert numpy.isclose(result.energy, -14.60266198, atol=1.e-3)

    U = mol.make_upccgsd_ansatz(name="SPA-UpCCGSD") # in this case no difference to SPA-UpCCSD
    E = tq.ExpectationValue(H=H, U=U)
    assert (len(E.extract_variables()) == 4)
    result = tq.minimize(E)
    assert numpy.isclose(result.energy, -14.60266198, atol=1.e-3)

    U = mol.make_upccgsd_ansatz(name="UpCCSD")
    E = tq.ExpectationValue(H=H, U=U)
    print(U)
    assert (len(E.extract_variables()) == 8)
    result = tq.minimize(E)
    assert numpy.isclose(result.energy, oigawert, atol=1.e-3)

    U = mol.make_upccgsd_ansatz(name="UpCCGSD")
    E = tq.ExpectationValue(H=H, U=U)
    assert (len(E.extract_variables()) == 12)
    result = tq.minimize(E)
    assert numpy.isclose(result.energy, oigawert, atol=1.e-3)

    U = mol.make_upccgsd_ansatz(name="UpCCGSD", direct_compiling=False)
    E = tq.ExpectationValue(H=H, U=U)
    assert (len(E.extract_variables()) == 12)
    result = tq.minimize(E)
    assert numpy.isclose(result.energy, oigawert, atol=1.e-3)

@pytest.mark.skipif(not has_pyscf, reason="PySCF not installed")
@pytest.mark.skipif(executable is None and not os.path.isfile('balanced_be_gtensor.npy'), reason="madness not installed and no files found")
def test_madness_pyscf_bridge():
    mol = tq.Molecule(name="balanced_be", geometry="Be 0.0 0.0 0.0", n_pno=2, pno={"diagonal": True, "maxrank": 1}, )
    H = mol.make_hamiltonian()
    e1 = numpy.linalg.eigvalsh(H.to_matrix())[0]
    e2 = mol.compute_energy("fci")
    e3 = mol.compute_energy("ccsd")
    e4 = mol.compute_energy("ccsd(t)")
    assert numpy.isclose(e1, e2)
    # CCSD(T) and CCSD are not exact but close in this case
    # primarily testing a functioning interface and keywords here
    assert numpy.isclose(e1, e4, atol=1.e-3)
    assert numpy.isclose(e3, e4, atol=1.e-3)