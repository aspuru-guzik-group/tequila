import pytest
import numpy
import os
import tequila as tq

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
    assert (numpy.isclose(-14.614662051580321, result.energy, atol=1.e-5))


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
    U = mol.make_upccgsd_ansatz(name="HCB-UpCCGD", direct_compiling="ladder")
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(E)
    assert numpy.isclose(result.energy, -14.60307768, atol=1.e-3)

    U = mol.make_upccgsd_ansatz(name="HCB-PNO-UpCCD")
    print(U)
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(E)
    assert numpy.isclose(result.energy, -14.60266198, atol=1.e-3)

    H = mol.make_hamiltonian()
    U = mol.make_upccgsd_ansatz(name="PNO-UpCCD")
    print(U)
    E = tq.ExpectationValue(H=H, U=U)
    variables = result.variables
    if "bravyi" in trafo.lower():
        # signs of angles change in BK compared to JW-like HCB
        variables = {k: -v for k, v in variables.items()}
        print(variables)
    energy = tq.simulate(E, variables)
    result = tq.minimize(E)
    assert numpy.isclose(result.energy, -14.60266198, atol=1.e-3)
    assert numpy.isclose(energy, -14.60266198, atol=1.e-3)

    U = mol.make_upccgsd_ansatz(name="PNO-UpCCSD")
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(E)
    assert numpy.isclose(result.energy, -14.60266198, atol=1.e-3)

    U = mol.make_upccgsd_ansatz(name="PNO-UpCCGSD")
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(E)
    assert numpy.isclose(result.energy, -14.60266198, atol=1.e-3)

    U = mol.make_upccgsd_ansatz(name="UpCCSD")
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(E)
    assert numpy.isclose(result.energy, -14.60266198, atol=1.e-3)

    U = mol.make_upccgsd_ansatz(name="UpCCGSD")
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(E)
    assert numpy.isclose(result.energy, -14.60307768, atol=1.e-3)

    U = mol.make_upccgsd_ansatz(name="UpCCGSD", direct_compiling=False)
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(E)
    assert numpy.isclose(result.energy, -14.60307768, atol=1.e-3)


@pytest.mark.parametrize("trafo", ["JordanWigner", "BravyiKitaev", "BravyiKitaevTree", "ReorderedJordanWigner",
                                   "ReorderedBravyiKitaev"])
@pytest.mark.skipif(executable is None and not os.path.isfile('balanced_be_gtensor.npy'),
                    reason="madness was not found")
def test_madness_separated_objective(trafo):
    direct_compiling = numpy.random.choice([True, False, "ladder"], 1)[0]
    n_pno = 2
    if os.path.isfile('balanced_be_gtensor.npy'):
        n_pno = None

    mol = tq.Molecule(name="balanced_be", geometry="Be 0.0 0.0 0.0", n_pno=n_pno, pno={"diagonal": True, "maxrank": 1},
                      transformation=trafo)

    E = mol.make_separated_objective(direct_compiling=direct_compiling)
    E2 = tq.ExpectationValue(H=mol.make_hardcore_boson_hamiltonian(), U=mol.make_upccgsd_ansatz(name="HCB-PNO-UpCCD", direct_compiling=direct_compiling))

    dE1 = tq.grad(E)
    dE2 = tq.grad(E2)
    for x in range(10):
        variables = {k:numpy.random.uniform(-2.0*numpy.pi, 2.0*numpy.pi,1)[0] for k in E.extract_variables()}
        f1 = tq.simulate(E, variables=variables)
        f2 = tq.simulate(E2, variables=variables)
        assert numpy.isclose(f1, f2, atol=1.e-4)
        k = numpy.random.choice(E.extract_variables())
        df1 = tq.simulate(dE1[k], variables=variables)
        df2 = tq.simulate(dE2[k], variables=variables)
        assert numpy.isclose(df1, df2, atol=1.e-4)

    result = tq.minimize(E)
    # for some reason it looses a bit of precision
    assert numpy.isclose(result.energy, -14.601923942565918)


@pytest.mark.parametrize("trafo", ["JordanWigner", "BravyiKitaev", "BravyiKitaevTree", "ReorderedJordanWigner",
                                   "ReorderedBravyiKitaev"])
@pytest.mark.skipif(executable is None and not os.path.isfile('balanced_be_gtensor.npy'),
                    reason="madness was not found")
def test_madness_separated_objective_active_space(trafo):
    n_pno = 2
    if os.path.isfile('balanced_be_gtensor.npy'):
        n_pno = None

    mol = tq.Molecule(active_orbitals=[1, 2], name="balanced_be", geometry="Be 0.0 0.0 0.0", n_pno=n_pno,
                      pno={"diagonal": True, "maxrank": 1},
                      transformation=trafo)

    E = mol.make_separated_objective(direct_compiling=True)
    result = tq.minimize(E)
    assert numpy.isclose(result.energy, -14.58913708)

    E2 = tq.ExpectationValue(H=mol.make_hardcore_boson_hamiltonian(), U=mol.make_upccgsd_ansatz(name="HCB-PNO-UpCCD", direct_compiling=True))
    result2 = tq.minimize(E2)
    assert numpy.isclose(result.energy, result2.energy, atol=1.e-4)

    for x in range(10):
        variables = {k:numpy.random.uniform(-2.0*numpy.pi, 2.0*numpy.pi,1)[0] for k in E.extract_variables()}
        f1 = tq.simulate(E, variables=variables)
        f2 = tq.simulate(E2, variables=variables)
        assert numpy.isclose(f1, f2, atol=1.e-4)
