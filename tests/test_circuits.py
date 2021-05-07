from tequila.circuit.gates import X, Y, Z, Rx, Ry, Rz, H, CNOT, QCircuit, RotationGate, Phase, ExpPauli, Trotterized, \
                                  U, u1, u2, u3, S, T, SWAP
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila.circuit._gates_impl import RotationGateImpl
from tequila.objective.objective import Variable
from tequila.simulators.simulator_api import simulate
from tequila import assign_variable, paulis
import numpy, sympy
import pytest


def test_qubit_map():

    for gate in [Rx, Ry, Rz]:
        U1 = gate(angle="a", target=0, control=1)
        U2 = gate(angle="a", target=1, control=0)
        mapped = U1.map_qubits(qubit_map={0:1, 1:0})
        assert len(mapped.gates) == len(U2.gates)
        for i in range(len(mapped.gates)):
            for k, v in mapped.gates[i].__dict__.items():
                assert U2.gates[i].__dict__[k] == v

    for gate in [H, X, Y, Z]:
        U1 = gate(target=0, control=1)
        U2 = gate(target=1, control=0)
        mapped = U1.map_qubits(qubit_map={0:1, 1:0})
        assert len(mapped.gates) == len(U2.gates)
        for i in range(len(mapped.gates)):
            for k, v in mapped.gates[i].__dict__.items():
                assert U2.gates[i].__dict__[k] == v

    for gate in [ExpPauli]:
        U1 = gate(angle="a", paulistring="X(0)Y(2)Z(3)", control=1)
        U2 = gate(angle="a", paulistring="X(5)Y(6)Z(7)", control=0)
        mapped = U1.map_qubits(qubit_map={0:5, 1:0, 2:6, 3:7})
        assert len(mapped.gates) == len(U2.gates)
        for i in range(len(mapped.gates)):
            for k, v in mapped.gates[i].__dict__.items():
                assert U2.gates[i].__dict__[k] == v

    for gate in [Trotterized]:
        U1 = gate(generator=paulis.X(0) + paulis.Y(0)*paulis.Z(1), angle="a", control=2, steps=1)
        U1 += gate(generator=paulis.Z(3)*paulis.Z(4), angle="b", control=2, steps=1)
        U2 = gate(generator=paulis.X(1) + paulis.Y(1)*paulis.Z(2), angle="a", control=0, steps=1)
        U2 += gate(generator=paulis.Z(4)*paulis.Z(5), angle="b", control=0)
        mapped = U1.map_qubits(qubit_map={0:1, 1:2, 3:4, 4:5, 2:0})
        assert len(mapped.gates) == len(U2.gates)
        for i in range(len(mapped.gates)):
            for k, v in mapped.gates[i].__dict__.items():
                assert U2.gates[i].__dict__[k] == v

def test_conventions():
    qubit = numpy.random.randint(0, 3)
    angle = Variable("angle")

    Rx1 = Rx(target=qubit, angle=angle)
    Rx2 = QCircuit.wrap_gate(RotationGateImpl(axis="X", target=qubit, angle=angle))
    Rx3 = QCircuit.wrap_gate(RotationGateImpl(axis="x", target=qubit, angle=angle))
    Rx4 = RotationGate(axis=0, target=qubit, angle=angle)
    Rx5 = RotationGate(axis="X", target=qubit, angle=angle)
    Rx6 = RotationGate(axis="x", target=qubit, angle=angle)
    Rx7 = RotationGate(axis=0, target=qubit, angle=angle)
    Rx7.axis = "X"

    ll = [Rx1, Rx2, Rx3, Rx4, Rx5, Rx6, Rx7]
    for l1 in ll:
        for l2 in ll:
            assert (l1 == l2)

    qubit = 2
    for c in [None, 0, 3]:
        for angle in ["angle", 0, 1.234]:
            for axes in [[0, "x", "X"], [1, "y", "Y"], [2, "z", "Z"]]:
                ll = [RotationGate(axis=i, target=qubit, control=c, angle=angle) for i in axes]
                for l1 in ll:
                    for l2 in ll:
                        assert (l1 == l2)
                        l1.axis = axes[numpy.random.randint(0, 2)]
                        assert (l1 == l2)


def strip_sympy_zeros(wfn: QubitWaveFunction):
    result = QubitWaveFunction()
    for k, v in wfn.items():
        if v != 0:
            result[k] = v
    return result


def test_basic_gates():
    I = sympy.I
    cos = sympy.cos
    sin = sympy.sin
    exp = sympy.exp
    BS = QubitWaveFunction.from_int
    angle = sympy.pi
    gates = [X(0), Y(0), Z(0), Rx(target=0, angle=angle), Ry(target=0, angle=angle), Rz(target=0, angle=angle), H(0)]
    results = [
        BS(1),
        I * BS(1),
        BS(0),
        cos(-angle / 2) * BS(0) + I * sin(-angle / 2) * BS(1),
        cos(-angle / 2) * BS(0) + I * sin(-angle / 2) * I * BS(1),
        exp(-I * angle / 2) * BS(0),
        1 / sympy.sqrt(2) * (BS(0) + BS(1))
    ]
    for i, g in enumerate(gates):
        wfn = simulate(g, backend="symbolic", variables={angle: numpy.pi})
        assert wfn.isclose(strip_sympy_zeros(results[i]))


def test_consistency():
    angle = numpy.pi / 2
    cpairs = [
        (CNOT(target=0, control=1), X(target=0, control=1)),
        (Ry(target=0, angle=numpy.pi), Rz(target=0, angle=4 * numpy.pi) + X(target=0)),
        (Rz(target=0, angle=numpy.pi), Rz(target=0, angle=numpy.pi) + Z(target=0)),
        (Rz(target=0, angle=angle), Rz(target=0, angle=angle / 2) + Rz(target=0, angle=angle / 2)),
        (Rx(target=0, angle=angle), Rx(target=0, angle=angle / 2) + Rx(target=0, angle=angle / 2)),
        (Ry(target=0, angle=angle), Ry(target=0, angle=angle / 2) + Ry(target=0, angle=angle / 2))
    ]

    for c in cpairs:
        print("circuit=", c[0], "\n", c[1])
        wfn1 = simulate(c[0], backend="symbolic")
        wfn2 = simulate(c[1], backend="symbolic")
        assert (numpy.isclose(wfn1.inner(wfn2), 1.0))


def test_moments():
    c = QCircuit()
    c += CNOT(target=0, control=(1, 2, 3))
    c += H(target=[0, 1])
    c += Rx(angle=numpy.pi, target=[0, 3])
    c += Z(target=1)
    c += Phase(phi=numpy.pi, target=4)
    moms = c.moments
    assert len(moms) == 3
    assert (moms[0].gates[1].parameter == assign_variable(numpy.pi))
    assert (moms[0].gates[1].target == (4,))


def test_canonical_moments():
    c = QCircuit()
    c += CNOT(target=0, control=(1, 2, 3))
    c += Rx(angle=Variable('a'), target=[0, 3])
    c += H(target=[0, 1])
    c += Rx(angle=Variable('a'), target=[2, 3])
    c += Rx(angle=Variable('a'), target=[0, 3])
    c += Z(target=1)
    c += Phase(phi=numpy.pi, target=4)
    moms = c.canonical_moments
    assert len(moms) == 6
    assert (moms[0].gates[1].parameter == assign_variable(numpy.pi))
    assert (moms[0].gates[1].target == (4,))
    assert hasattr(moms[3].gates[0], 'axis')
    assert len(moms[0].qubits) == 5


def test_circuit_from_moments():
    c = QCircuit()
    c += CNOT(target=0, control=(1, 2, 3))
    c += Phase(phi=numpy.pi, target=4)
    c += Rx(angle=Variable('a'), target=[0, 3])
    c += H(target=[0, 1])
    c += Rx(angle=Variable('a'), target=[2, 3])
    ## table[1] should equal 1 at this point, len(moments should be 3)
    c += Z(target=1)
    c += Rx(angle=Variable('a'), target=[0, 3])
    moms = c.moments
    c2 = QCircuit.from_moments(moms)
    assert c == c2


@pytest.mark.parametrize(
    "gate, angle",
    [
        (Z(target=0, control=None), numpy.pi),              # Z = u1(pi)
        (Z(target=0, control=1), numpy.pi),
        (S(target=0, control=None), numpy.pi/2),            # S = u1(pi/2)
        (S(target=0, control=1), numpy.pi/2),
        (S(target=0, control=None).dagger(), -numpy.pi/2),  # Sdg = u1(-pi/2)
        (S(target=0, control=1).dagger(), -numpy.pi/2),
        (T(target=0, control=None), numpy.pi/4),            # T = u1(pi/4)
        (T(target=0, control=1), numpy.pi/4),
        (T(target=0, control=None).dagger(), -numpy.pi/4),  # Tdg = u1(-pi/4)
        (T(target=0, control=1).dagger(), -numpy.pi/4)
    ]
)
def test_unitary_gate_u1(gate, angle):
    """
    Test some equivalences for u1 gate
    """
    c_u1 = u1(lambd=angle, target=gate.gates[0].target,
              control=None if len(gate.gates[0].control) == 0 else gate.gates[0].control)

    if len(gate.gates[0].control) > 0:
        c_u1 = X(target=gate.gates[0].control) + c_u1
        gate = X(target=gate.gates[0].control) + gate

    wfn1 = simulate(c_u1, backend="symbolic")
    wfn2 = simulate(gate, backend="symbolic")

    assert (numpy.isclose(wfn1.inner(wfn2), 1.0))


@pytest.mark.parametrize(
    "ctrl, phi, lambd",
    [
        (None, numpy.pi / 13, numpy.pi / 7),
        (1, numpy.pi / 13, numpy.pi / 7),
        (None, 0, 0),
        (1, 0, 0),
        (None, numpy.pi, numpy.pi),
        (1, numpy.pi, numpy.pi),
        (None, 0, numpy.pi),
        (1, numpy.pi, 0),
    ]
)
def test_unitary_gate_u2(ctrl, phi, lambd):
    """
    Test some equivalences for u2 gate
    Since u2(\\phi, \\lambda) = Rz(\\phi)Ry(\\pi/2)Rz(\\lambda)
    """
    c_u2 = u2(phi=phi, lambd=lambd, target=0, control=ctrl)
    c_equiv = Rz(target=0, control=ctrl, angle=lambd) + \
              Ry(target=0, control=ctrl, angle=numpy.pi / 2) + \
              Rz(target=0, control=ctrl, angle=phi)

    if ctrl is not None:
        c_u2 = X(target=ctrl) + c_u2
        c_equiv = X(target=ctrl) + c_equiv

    wfn1 = simulate(c_u2, backend="symbolic")
    wfn2 = simulate(c_equiv, backend="symbolic")

    assert (numpy.isclose(wfn1.inner(wfn2), 1.0))


@pytest.mark.parametrize(
    "gate, theta, phi, lambd",
    [
        (Rx(target=0, control=None, angle=numpy.pi/5), numpy.pi/5, -numpy.pi/2, numpy.pi/2),  # Rx(angle) = u3(angle, -pi/2, pi/2)
        (Rx(target=0, control=1, angle=numpy.pi/6), numpy.pi/6, -numpy.pi/2, numpy.pi/2),
        (Rx(target=0, control=None, angle=numpy.pi/7), numpy.pi/7, -numpy.pi/2, numpy.pi/2),
        (Rx(target=0, control=1, angle=numpy.pi/8), numpy.pi/8, -numpy.pi/2, numpy.pi/2),
        (Ry(target=0, control=1, angle=numpy.pi/4), numpy.pi/4, 0, 0),                        # Ry(angle) = u3(angle, 0, 0)
        (Ry(target=0, control=1, angle=numpy.pi/5), numpy.pi/5, 0, 0),
        (Ry(target=0, control=1, angle=numpy.pi/3), numpy.pi/3, 0, 0),
        (Ry(target=0, control=1, angle=numpy.pi/2), numpy.pi/2, 0, 0),
        (Rz(target=0, control=None, angle=numpy.pi), 0, 0, numpy.pi),                         # Rz(angle) = U(0, 0, angle)
        (Rz(target=0, control=1, angle=numpy.pi/6), 0, 0, numpy.pi/6),
        (Rz(target=0, control=None, angle=numpy.pi/7), 0, 0, numpy.pi/7),
        (Rz(target=0, control=1, angle=numpy.pi/8), 0, 0, numpy.pi/8)
    ]
)
def test_unitary_gate_u_u3(gate, theta, phi, lambd):
    """
    Test some equivalences for u3 gate (also U gate, because U = u3)
    """
    c_u3 = u3(theta=theta, phi=phi, lambd=lambd, target=gate.gates[0].target,
              control=None if len(gate.gates[0].control) == 0 else gate.gates[0].control)

    if len(gate.gates[0].control) > 0:
        c_u3 = X(target=gate.gates[0].control) + c_u3
        gate = X(target=gate.gates[0].control) + gate

    wfn1 = simulate(c_u3, backend="symbolic")
    wfn2 = simulate(gate, backend="symbolic")

    assert (numpy.isclose(wfn1.inner(wfn2), 1.0))

def test_swap():
    U = X(0)
    U += SWAP(0,2)
    wfn = simulate(U)
    wfnx = simulate(X(2))
    assert numpy.isclose(numpy.abs(wfn.inner(wfnx))**2,1.0)

    U = X(2)
    U += SWAP(0,2, power=2.0)
    wfn = simulate(U)
    wfnx = simulate(X(0))
    assert numpy.isclose(numpy.abs(wfn.inner(wfnx))**2,1.0)

    U = X(0)+X(3)
    U += SWAP(0,2, control=3)
    wfn = simulate(U)
    wfnx = simulate(X(2)+X(3))
    assert numpy.isclose(numpy.abs(wfn.inner(wfnx))**2,1.0)

    U = X(2)+X(3)
    U += SWAP(0,2, control=3, power=2.0)
    wfn = simulate(U)
    wfnx = simulate(X(2)+X(3))
    assert numpy.isclose(numpy.abs(wfn.inner(wfnx))**2,1.0)





