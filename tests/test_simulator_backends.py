"""
All Backends need to be installed for full testing
"""

import numpy
import pytest
import random
import numbers
import tequila as tq
import tequila.simulators.simulator_api

"""
Warn if Simulators are not installed
"""
import warnings
import os, glob


def teardown_function(function):
    [os.remove(x) for x in glob.glob("*.npy")]
    [os.remove(x) for x in glob.glob("qvm.log")]
    [os.remove(x) for x in glob.glob("*.dat")]


@pytest.mark.dependencies
def test_dependencies():
    for package in tequila.simulators.simulator_api.SUPPORTED_BACKENDS:
        if package != "qulacs_gpu":
            assert (package in tq.simulators.simulator_api.INSTALLED_BACKENDS)


@pytest.mark.parametrize("backend", list(set(
    [None] + [k for k in tequila.simulators.simulator_api.INSTALLED_SIMULATORS.keys()] + [k for k in
                                                                                          tequila.simulators.simulator_api.INSTALLED_SAMPLERS.keys()])))
def test_interface(backend):
    H = tq.paulis.X(0)
    U = tq.gates.X(target=0)
    CU = tq.compile(objective=U, backend=backend)
    a = tq.simulate(objective=U, backend=backend)
    aa = CU()
    aaa = tq.compile_to_function(objective=U, backend=backend)()
    assert (isinstance(a, tq.QubitWaveFunction))
    assert (aa.isclose(a))
    assert (aaa.isclose(a))
    E = tq.ExpectationValue(H=H, U=U)
    CE = tq.compile(objective=E, backend=backend)
    a = tq.simulate(objective=E, backend=backend)
    aa = CE()
    aaa = tq.compile_to_function(objective=E, backend=backend)()

    assert (isinstance(a, numbers.Number))
    assert (aa == a)
    assert (aaa == a)


INSTALLED_SIMULATORS = tequila.simulators.simulator_api.INSTALLED_SIMULATORS.keys()
INSTALLED_SAMPLERS = tequila.simulators.simulator_api.INSTALLED_SAMPLERS.keys()

@pytest.mark.parametrize("backend", INSTALLED_SAMPLERS)
def test_sampling_accumulation(backend):
    # minimal test that was added after a bug was discovered
    # just needs to asssure that it runs through and no errors are thrown within the process
    U = tq.gates.Ry(angle=numpy.pi/2, target=0) + tq.gates.CNOT(1,3)
    H = tq.paulis.Qm(1)
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.simulate(E, backend=backend, samples=100)
    assert result == 0.0


@pytest.mark.parametrize("backend", INSTALLED_SAMPLERS)
def test_sampling_accumulation(backend):
    # minimal test that was added after a bug was discovered
    # just needs to asssure that it runs through and no errors are thrown within the process
    U = tq.gates.Ry(angle=numpy.pi / 2, target=0) + tq.gates.CNOT(1, 3)
    H = tq.paulis.Qm(1)
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.simulate(E, backend=backend, samples=100)
    assert result == 0.0


@pytest.mark.parametrize("backend", INSTALLED_SAMPLERS)
def test_sampling_circuits(backend):
    U = tq.gates.X([1, 3, 5])
    U += tq.gates.X([0, 2, 4, 6])
    U += tq.gates.X([1, 3, 5])

    d1 = tq.simulate(U, samples=10, backend=backend)
    assert d1[1 + 4 + 16 + 64] == 10

    d1 = tq.simulate(U, samples=10, backend=backend, read_out_qubits=[1, 3, 5])
    assert d1[0] == 10

    d1 = tq.simulate(U, samples=10, backend=backend, read_out_qubits=[0, 2, 4, 6])
    if 1 + 2 + 4 + 8 not in d1 or d1[1 + 2 + 4 + 8] != 10:
        print("got this = ", d1)
    print("got this = ", d1)
    assert d1[1 + 2 + 4 + 8] == 10


@pytest.mark.parametrize("backend", INSTALLED_SAMPLERS)
def test_sampling_expvals(backend):
    U = tq.gates.X([0, 1, 2])
    H = tq.paulis.Z(0)
    E1 = tq.simulate(tq.ExpectationValue(H=H, U=U), backend=backend, samples=1000)
    assert numpy.isclose(E1, -1.0)
    H = tq.paulis.Z([0, 1])
    E1 = tq.simulate(tq.ExpectationValue(H=H, U=U), backend=backend, samples=1000)
    assert numpy.isclose(E1, 1.0)
    H = tq.paulis.Z([0, 1, 2])
    E1 = tq.simulate(tq.ExpectationValue(H=H, U=U), backend=backend, samples=1000)
    assert numpy.isclose(E1, -1.0)

    U = tq.gates.H([0, 1, 2])
    H = tq.paulis.X(0)
    E1 = tq.simulate(tq.ExpectationValue(H=H, U=U), backend=backend, samples=1000)
    assert numpy.isclose(E1, 1.0)
    H = tq.paulis.X([0, 1])
    E1 = tq.simulate(tq.ExpectationValue(H=H, U=U), backend=backend, samples=1000)
    assert numpy.isclose(E1, 1.0)
    H = tq.paulis.X([0, 1, 2])
    E1 = tq.simulate(tq.ExpectationValue(H=H, U=U), backend=backend, samples=1000)
    assert numpy.isclose(E1, 1.0)

    U = tq.gates.H([0, 1, 2])
    H = tq.paulis.Zero()
    E1 = tq.simulate(tq.ExpectationValue(H=H, U=U), backend=backend, samples=1000)
    assert numpy.isclose(E1, 0.0)
    U = tq.gates.H([0, 1, 2])
    H = tq.paulis.X(3) + 1.234 * tq.paulis.I()
    E1 = tq.simulate(tq.ExpectationValue(H=H, U=U), backend=backend, samples=1000)
    assert numpy.isclose(E1, 1.234)


@pytest.mark.parametrize("backend",
                         list(set([None] + [k for k in INSTALLED_SIMULATORS] + [k for k in INSTALLED_SAMPLERS])))
@pytest.mark.parametrize("samples", [None, 10])
def test_parametrized_interface(backend, samples):
    if samples is not None and backend not in INSTALLED_SAMPLERS:
        pytest.skip("sampling not yet supported for backend={}".format(backend))

    H = tq.paulis.X(0)
    U = tq.gates.Ry(angle="a", target=0)
    variables = {"a": numpy.pi / 2}
    CU = tq.compile(objective=U, backend=backend, samples=None)
    a = tq.simulate(objective=U, backend=backend, variables=variables, samples=None)
    aa = CU(variables=variables, samples=None)
    aaa = tq.compile_to_function(objective=U, backend=backend, samples=samples)(variables["a"], samples=None)
    assert (isinstance(a, tq.QubitWaveFunction))
    assert (aa.isclose(a))
    assert (aaa.isclose(a))
    E = tq.ExpectationValue(H=H, U=U)
    CE = tq.compile(objective=E, backend=backend, samples=samples)
    a = tq.simulate(objective=E, backend=backend, variables=variables, samples=samples)
    aa = CE(variables=variables, samples=samples)
    aaa = tq.compile_to_function(objective=E, backend=backend)(variables["a"], samples=samples)

    assert (isinstance(a, numbers.Number))
    assert numpy.isclose(aa, a, 1.e-1)
    assert numpy.isclose(aaa, a, 1.e-1)


@pytest.mark.parametrize("name", tequila.simulators.simulator_api.SUPPORTED_BACKENDS)
def test_backend_availability(name):
    if name not in tq.INSTALLED_BACKENDS:
        warnings.warn(name + " is not installed!", UserWarning)


@pytest.mark.parametrize("simulator", tequila.simulators.simulator_api.INSTALLED_SIMULATORS.keys())
@pytest.mark.parametrize("angle", numpy.random.uniform(0.0, 2.0 * numpy.pi, 2))
def test_rotations(simulator, angle):
    U1 = tq.gates.X(target=1) + tq.gates.X(target=0, control=1) + tq.gates.Rx(angle=angle, target=0)
    U2 = tq.gates.X(target=1) + tq.gates.X(target=0, control=1) + tq.gates.ExpPauli(angle=angle, paulistring="X(0)")
    wfn1 = tequila.simulators.simulator_api.simulate(U1, backend=None)
    wfn2 = tequila.simulators.simulator_api.simulate(U2, backend=None)
    wfn3 = tequila.simulators.simulator_api.simulate(U2, backend=simulator)
    wfn4 = tequila.simulators.simulator_api.simulate(U2, backend=simulator)

    assert (numpy.isclose(numpy.abs(wfn1.inner(wfn2)) ** 2, 1.0, atol=1.e-4))
    assert (numpy.isclose(numpy.abs(wfn3.inner(wfn4)) ** 2, 1.0, atol=1.e-4))
    assert (numpy.isclose(numpy.abs(wfn1.inner(wfn3)) ** 2, 1.0, atol=1.e-4))

    U = tq.gates.X(target=1) + tq.gates.X(target=0, control=1) + tq.gates.ExpPauli(angle=angle, paulistring="X(0)Y(3)")
    wfn1 = tq.simulate(U2, backend=None)
    wfn2 = tq.simulate(U2, backend=simulator)

    assert (numpy.isclose(numpy.abs(wfn1.inner(wfn2)) ** 2, 1.0, atol=1.e-4))


@pytest.mark.parametrize("simulator", tequila.simulators.simulator_api.INSTALLED_SIMULATORS.keys())
@pytest.mark.parametrize("angle", numpy.random.uniform(0.0, 2.0 * numpy.pi, 2))
@pytest.mark.parametrize("ps", ["X(0)Y(3)",
                                "Y(2)X(4)"])  # it is important to test paulistrings on qubits which are not explicitly initialized through other gates
def test_multi_pauli_rotation(simulator, angle, ps):
    U = tq.gates.X(target=1) + tq.gates.X(target=0, control=1) + tq.gates.ExpPauli(angle=angle, paulistring=ps)
    wfn1 = tequila.simulators.simulator_api.simulate(U, backend=None)
    wfn2 = tequila.simulators.simulator_api.simulate(U, backend=simulator)

    assert (numpy.isclose(numpy.abs(wfn1.inner(wfn2)) ** 2, 1.0, atol=1.e-4))


@pytest.mark.parametrize("simulator", tequila.simulators.simulator_api.INSTALLED_SIMULATORS.keys())
@pytest.mark.parametrize("angle", numpy.random.uniform(0.0, 2.0 * numpy.pi, 2))
def test_parametrized_rotations(simulator, angle):
    U1 = tq.gates.X(target=1) + tq.gates.X(target=0, control=1) + tq.gates.Rx(angle="a", target=0)
    U2 = tq.gates.X(target=1) + tq.gates.X(target=0, control=1) + tq.gates.ExpPauli(angle="a", paulistring="X(0)")
    variables = {"a": angle}
    wfn1 = tequila.simulators.simulator_api.simulate(U1, variables, backend=None)
    wfn2 = tequila.simulators.simulator_api.simulate(U2, variables, backend=None)
    wfn3 = tequila.simulators.simulator_api.simulate(U2, variables, backend=simulator)
    wfn4 = tequila.simulators.simulator_api.simulate(U2, variables, backend=simulator)

    assert (numpy.isclose(numpy.abs(wfn1.inner(wfn2)) ** 2, 1.0, atol=1.e-4))
    assert (numpy.isclose(numpy.abs(wfn3.inner(wfn4)) ** 2, 1.0, atol=1.e-4))
    assert (numpy.isclose(numpy.abs(wfn1.inner(wfn3)) ** 2, 1.0, atol=1.e-4))


@pytest.mark.parametrize("simulator", tequila.simulators.simulator_api.INSTALLED_SIMULATORS.keys())
@pytest.mark.parametrize("angle", numpy.random.uniform(0.0, 2.0 * numpy.pi, 2))
@pytest.mark.parametrize("ps", ["X(0)Z(3)", "Y(2)X(4)"])
def test_parametrized_multi_pauli_rotation(simulator, angle, ps):
    a = tq.Variable("a")
    variables = {a: angle}
    U = tq.gates.X(target=1) + tq.gates.X(target=0, control=1) + tq.gates.ExpPauli(angle=a, paulistring=ps)
    wfn1 = tequila.simulators.simulator_api.simulate(U, variables, backend=None)
    wfn2 = tequila.simulators.simulator_api.simulate(U, variables, backend=simulator)
    assert (numpy.isclose(numpy.abs(wfn1.inner(wfn2)) ** 2, 1.0, atol=1.e-4))


def create_random_circuit():
    primitive_gates = [tq.gates.X, tq.gates.Y, tq.gates.Z, tq.gates.H]
    rot_gates = [tq.gates.Rx, tq.gates.Ry, tq.gates.Rz]
    circuit = tq.gates.QCircuit()
    for x in range(4):
        target = random.randint(1, 2)
        control = random.randint(3, 4)
        circuit += random.choice(primitive_gates)(target=target, control=control)
        target = random.randint(1, 2)
        control = random.randint(3, 4)
        angle = random.uniform(0.0, 4.0)
        circuit += random.choice(rot_gates)(target=target, control=control, angle=angle)
    return circuit


@pytest.mark.parametrize("simulator", tequila.simulators.simulator_api.INSTALLED_SIMULATORS.keys())
def test_wfn_simple_execution(simulator):
    ac = tq.gates.X(0)
    ac += tq.gates.Ry(target=1, control=0, angle=2.3 / 2)
    ac += tq.gates.H(target=1, control=None)
    tequila.simulators.simulator_api.simulate(ac, backend=simulator)


@pytest.mark.parametrize("simulator", tequila.simulators.simulator_api.INSTALLED_SIMULATORS.keys())
def test_wfn_multitarget(simulator):
    ac = tq.gates.X([0, 1, 2])
    ac += tq.gates.Ry(target=[1, 2], control=0, angle=2.3 / 2)
    ac += tq.gates.H(target=[1], control=None)
    tequila.simulators.simulator_api.simulate(ac, backend=simulator)


@pytest.mark.parametrize("simulator", tequila.simulators.simulator_api.INSTALLED_SIMULATORS.keys())
def test_wfn_multi_control(simulator):
    # currently no compiler, so that test can not succeed
    if simulator == 'qiskit':
        return
    ac = tq.gates.X([0, 1, 2])
    ac += tq.gates.Ry(target=[0], control=[1, 2], angle=2.3 / 2)
    ac += tq.gates.H(target=[0], control=[1])
    tequila.simulators.simulator_api.simulate(ac, backend=simulator)
    ac = tq.gates.X([0, 1, 2])
    ac += tq.gates.Ry(target=[0], control=[1, 2], angle=2.3 / 2)
    ac += tq.gates.H(target=[0], control=[1, 2])
    tequila.simulators.simulator_api.simulate(ac, backend=simulator)


@pytest.mark.parametrize("simulator", tequila.simulators.simulator_api.INSTALLED_SIMULATORS.keys())
def test_wfn_simple_consistency(simulator):
    ac = tequila.circuit.QCircuit()
    for x in range(1, 5):
        ac += tq.gates.X(x)
    ac += create_random_circuit()
    print(ac)
    wfn0 = tequila.simulators.simulator_api.simulate(ac, backend=simulator)
    wfn1 = tequila.simulators.simulator_api.simulate(ac, backend=None)
    print(wfn0)
    print(wfn1)
    assert (wfn0.isclose(wfn1))


@pytest.mark.parametrize("simulator", tequila.simulators.simulator_api.INSTALLED_SAMPLERS.keys())
def test_shot_simple_execution(simulator):
    ac = tq.gates.X(0)
    ac += tq.gates.Ry(target=1, control=0, angle=1.2 / 2)
    ac += tq.gates.H(target=1, control=None)
    tequila.simulators.simulator_api.simulate(ac, backend=simulator,
                                              samples=1, pyquil_backend="2q-qvm",
                                              additional_keyword="Andreas-Dorn",
                                              read_out_qubits=[0, 1])


@pytest.mark.parametrize("simulator", tequila.simulators.simulator_api.INSTALLED_SAMPLERS.keys())
def test_shot_multitarget(simulator):
    ac = tq.gates.X([0, 1, 2])
    ac += tq.gates.Ry(target=[1, 2], control=0, angle=2.3 / 2)
    ac += tq.gates.H(target=[1], control=None)
    tequila.simulators.simulator_api.simulate(ac, backend=simulator, samples=1, read_out_qubits=[0, 1])


@pytest.mark.parametrize("simulator", tequila.simulators.simulator_api.INSTALLED_SAMPLERS.keys())
def test_shot_multi_control(simulator):
    ac = tq.gates.X([0, 1, 2])
    ac += tq.gates.X(target=[0], control=[1, 2])
    ac += tq.gates.Ry(target=[0], control=[1, 2], angle=2.3 / 2)
    ac += tq.gates.Rz(target=[0], control=[1, 2], angle=2.3 / 2)
    ac += tq.gates.Rx(target=[0], control=[1, 2], angle=2.3 / 2)
    tequila.simulators.simulator_api.simulate(ac, backend=simulator, samples=1, read_out_qubits=[0, 1])


@pytest.mark.skipif(condition='cirq' not in tq.INSTALLED_SAMPLERS or 'qiskit' not in tq.INSTALLED_SAMPLERS,
                    reason="need at least two samplers")
def test_shot_simple_consistency():
    samplers = tq.INSTALLED_SAMPLERS.keys()
    ac = create_random_circuit()
    reference = tequila.simulate(ac, backend=None, samples=1000)
    for sampler in samplers:
        wfn = tequila.simulate(ac, backend=sampler, samples=1000)
        if not reference.isclose(wfn):
            raise Exception("failed for {}\n{} vs \n{}".format(sampler, reference, wfn))
        assert reference.isclose(wfn)


@pytest.mark.parametrize("simulator", tequila.simulators.simulator_api.INSTALLED_SIMULATORS.keys())
@pytest.mark.parametrize("initial_state", numpy.random.randint(0, 31, 5))
def test_initial_state_from_integer(simulator, initial_state):
    U = tq.gates.QCircuit()
    for i in range(6):
        U += tq.gates.X(target=i) + tq.gates.X(target=i)

    wfn = tq.simulate(U, initial_state=initial_state, backend=simulator)
    assert (initial_state in wfn)
    assert (numpy.isclose(wfn[initial_state], 1.0))


@pytest.mark.parametrize("backend", tequila.simulators.simulator_api.INSTALLED_SIMULATORS.keys())
def test_hamiltonian_reductions(backend):
    for q in [0, 1, 2, 3, 4]:
        H = tq.paulis.Z(qubit=[0, 1, 2, 3, 4])
        U = tq.gates.X(target=q)
        U2 = tq.gates.X(target=q) + tq.gates.X(target=[0, 1, 2, 3, 4]) + tq.gates.X(target=[0, 1, 2, 3, 4])
        E1 = tq.compile(tq.ExpectationValue(H=H, U=U), backend=backend)
        E2 = tq.compile(tq.ExpectationValue(H=H, U=U2), backend=backend)
        assert E1.get_expectationvalues()[0]._reduced_hamiltonians[0] == tq.paulis.Z(q)
        assert numpy.isclose(E1(), E2())


@pytest.mark.parametrize("backend", tequila.simulators.simulator_api.INSTALLED_SAMPLERS.keys())
def test_sampling(backend):
    U = tq.gates.Ry(angle=0.0, target=0)
    H = tq.paulis.X(0)
    E = tq.ExpectationValue(H=H, U=U)

    for i in range(10):
        e = tq.simulate(E, samples=1000, backend=backend)
        assert numpy.isclose(e, 0.0, atol=2.e-1)

    E = tq.compile(E, backend=backend, samples=1000)
    for i in range(10):
        e = E(samples=1000)
        assert numpy.isclose(e, 0.0, atol=2.e-1)


@pytest.mark.parametrize("backend", tequila.simulators.simulator_api.INSTALLED_SAMPLERS.keys())
def test_sampling_read_out_qubits(backend):
    U = tq.gates.X(0)
    U += tq.gates.Z(1)

    wfn = tq.QubitWaveFunction(2)

    result = tq.simulate(U, backend=backend, samples=1, read_out_qubits=[0, 1])
    assert (numpy.isclose(numpy.abs(wfn.inner(result)) ** 2, 1.0, atol=1.e-4))

    result = tq.simulate(U, backend=backend, samples=1, read_out_qubits=[1, 0])
    assert (numpy.isclose(numpy.abs(wfn.inner(result)) ** 2, 1.0, atol=1.e-4))


@pytest.mark.parametrize("backend", INSTALLED_SIMULATORS)
def test_qubit_excitations(backend):
    if backend == "symbolic":
        return
    U1 = tq.gates.X(0) + tq.gates.QubitExcitation(angle=numpy.pi / 2, target=[0, 1])
    U2 = tq.gates.H(0) + tq.gates.X(1) + tq.gates.CNOT(0, 1)
    wfn1 = tq.simulate(U1, backend=backend)
    wfn2 = tq.simulate(U2, backend=backend)
    F = numpy.abs(wfn1.inner(wfn2)) ** 2
    assert numpy.isclose(F, 1.0, 1.e-4)

    U1 = tq.gates.X([0, 1]) + tq.gates.QubitExcitation(angle=numpy.pi / 2, target=[0, 2, 1, 3])
    U2 = tq.gates.H(0) + tq.gates.X([2, 3]) + tq.gates.CNOT(0, 1) + tq.gates.CNOT(0, 2) + tq.gates.CNOT(0, 3)
    wfn1 = tq.simulate(U1, backend=backend)
    wfn2 = tq.simulate(U2, backend=backend)
    F = numpy.abs(wfn1.inner(wfn2)) ** 2
    assert numpy.isclose(F, 1.0, 1.e-4)

    U1 = tq.gates.X([0, 1, 2, 3]) + tq.gates.QubitExcitation(angle=numpy.pi / 2, target=[0, 4, 1, 5, 2, 6, 3, 7])
    U2 = tq.gates.H(0) + tq.gates.X([4, 5, 6, 7]) + tq.gates.X([1, 2, 3], 0) + tq.gates.X([4, 5, 6, 7], 0)
    wfn1 = tq.simulate(U1, backend=backend)
    wfn2 = tq.simulate(U2, backend=backend)
    F = numpy.abs(wfn1.inner(wfn2)) ** 2
    print(wfn1)
    print(wfn2)
    print(tq.compile(U1, backend=backend).abstract_circuit)
    assert numpy.isclose(F, 1.0, 1.e-4)

    q = [5, 3, 7, 8, 2, 9, 2, 4]

    U1 = tq.gates.X([q[0], q[1], q[2]]) + tq.gates.QubitExcitation(angle=numpy.pi / 2,
                                                                   target=[q[0], q[3], q[1], q[4], q[2], q[5]])
    U2 = tq.gates.H(q[0]) + tq.gates.X([q[3], q[4], q[5]]) + tq.gates.X([q[1], q[2]], q[0]) + tq.gates.X(
        [q[3], q[4], q[5]], q[0])
    wfn1 = tq.simulate(U1, backend=backend)
    wfn2 = tq.simulate(U2, backend=backend)
    F = numpy.abs(wfn1.inner(wfn2)) ** 2

    assert numpy.isclose(F, 1.0, 1.e-4)