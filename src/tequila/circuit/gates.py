from tequila.circuit.circuit import QCircuit
from tequila.objective.objective import Variable, assign_variable
from tequila.circuit import _gates_impl as impl
import typing, numbers
from tequila.hamiltonian import PauliString, QubitHamiltonian, paulis
from tequila.tools import list_assignment
import numpy as np
import functools
import copy


def Phase(target: typing.Union[list, int],
          control: typing.Union[list, int] = None, angle: typing.Union[typing.Hashable, numbers.Number] = None, *args,
          **kwargs) -> QCircuit:
    """
    Notes
    ----------
    Initialize an abstract phase gate which acts as

    .. math::
        S(\\phi) = \\begin{pmatrix} 1 & 0 \\\\ 0 & e^{i\\phi} \\end{pmatrix}

    Parameters
    ----------
    angle
        defines the phase, can be numeric type (static gate) or hashable non-numeric type (parametrized gate)
    target
        int or list of int
    control
        int or list of int

    Returns
    -------
    QCircuit object

    """

    # ensure backward compatibility
    if "phi" in kwargs:
        if angle is None:
            angle = kwargs["phi"]
        else:
            raise Exception(
                "tq.gates.Phase initialization: You gave two angles angle={} and phi={}. Please only use angle".format(
                    angle, kwargs["phi"]))

    if angle is None:
        angle = np.pi

    target = list_assignment(target)
    gates = [impl.PhaseGateImpl(phase=angle, target=q, control=control) for q in target]

    return QCircuit.wrap_gate(gates)


def S(target: typing.Union[list, int], control: typing.Union[list, int] = None) -> QCircuit:
    """
    Notes
    ----------

    .. math::
        S = \\begin{pmatrix} 1 & 0 \\\\ 0 & e^{i\\frac{\\pi}{2}} \\end{pmatrix}

    Parameters
    ----------
    target
        int or list of int
    control
        int or list of int

    Returns
    -------
    QCircuit object
    """
    return Phase(angle=np.pi / 2, target=target, control=control)


def T(target: typing.Union[list, int], control: typing.Union[list, int] = None):
    """
    Notes
    ----------
    Fixed phase gate

    .. math::
        T = \\begin{pmatrix} 1 & 0 \\\\ 0 & e^{i\\frac{\\pi}{4}} \\end{pmatrix}

    Parameters
    ----------
    target
        int or list of int
    control
        int or list of int

    Returns
    -------
    QCircuit object

    """
    return Phase(angle=np.pi / 4, target=target, control=control)

def Rx(angle, target: typing.Union[list, int], control: typing.Union[list, int] = None, assume_real=False) -> QCircuit:
    """
    Notes
    ----------
    Rx gate of the form

    .. math::
        R_{x}(\\text{angle}) = e^{-i\\frac{\\text{angle}}{2} \\sigma_{x}}


    Parameters
    ----------
    angle
        Hashable type (will be treated as Variable) or Numeric type (static angle)
    target
        integer or list of integers
    control
        integer or list of integers
    assume_real
        enable improved gradient compilation for controlled gates
    Returns
    -------
    QCircuit object with this RotationGate

    """
    return RotationGate(axis=0, angle=angle, target=target, control=control, assume_real=assume_real)


def Ry(angle, target: typing.Union[list, int], control: typing.Union[list, int] = None, assume_real=False) -> QCircuit:
    """
    Notes
    ----------
    Ry gate of the form

    .. math::
        R_{y}(\\text{angle}) = e^{-i\\frac{\\text{angle}}{2} \\sigma_{y}}


    Parameters
    ----------
    angle
        Hashable type (will be treated as Variable) or Numeric type (static angle)
    target
        integer or list of integers
    control
        integer or list of integers

    Returns
    -------
    QCircuit object with this RotationGate
    """
    return RotationGate(axis=1, angle=angle, target=target, control=control, assume_real=assume_real)


def Rz(angle, target: typing.Union[list, int], control: typing.Union[list, int] = None, assume_real=False) -> QCircuit:
    """
    Notes
    ----------
    Rz gate of the form

    .. math::
        R_{z}(\\text{angle}) = e^{-i\\frac{\\text{angle}}{2} \\sigma_{z}}


    Parameters
    ----------
    angle
        Hashable type (will be treated as Variable) or Numeric type (static angle)
    target
        integer or list of integers
    control
        integer or list of integers

    Returns
    QCircuit object with this RotationGate
    -------
    """
    return RotationGate(axis=2, angle=angle, target=target, control=control, assume_real=assume_real)


def X(target: typing.Union[list, int], control: typing.Union[list, int] = None, power=None, angle=None, *args, **kwargs) -> QCircuit:
    """
    Notes
    ----------
    Pauli X Gate

    Parameters
    ----------
    target
        int or list of int
    control
        int or list of int
    power
        numeric type (fixed exponent) or hashable type (parametrized exponent)
    angle
        similar to power, but will be interpreted as
        .. math::
           U(\\text{angle})=e^{-i\\frac{angle}{2} (1-X)}
        the default is angle=pi
        .. math::
           U(\\pi) = X
        If angle and power are given both, tequila will combine them

    Returns
    -------
    QCircuit object
    """

    generator = lambda q: paulis.X(q) - paulis.I(q)
    return _initialize_power_gate(name="X", power=power, angle=angle, target=target, control=control,
                                  generator=generator, *args, **kwargs)


def H(target: typing.Union[list, int], control: typing.Union[list, int] = None, power=None, angle=None, *args, **kwargs) -> QCircuit:
    """
    Notes
    ----------
    Hadamard gate

    Parameters
    ----------
    target
        int or list of int
    control
        int or list of int
    power
        numeric type (fixed exponent) or hashable type (parametrized exponent)
        angle
        similar to power, but will be interpreted as
        .. math::
           U(\\text{angle})=e^{-i\\frac{angle}{2} generator}
        the default is angle=pi
        .. math::
           U(\\pi) = H
        If angle and power are given both, tequila will combine them

    Returns
    -------
    QCircuit object

    """
    coef = 1 / np.sqrt(2)
    generator = lambda q: coef * (paulis.Z(q) + paulis.X(q)) - paulis.I(q)
    return _initialize_power_gate(name="H", power=power, angle=angle, target=target, control=control,
                                  generator=generator, *args, **kwargs)


def Y(target: typing.Union[list, int], control: typing.Union[list, int] = None, power=None, angle=None, *args, **kwargs) -> QCircuit:
    """
    Notes
    ----------
    Pauli Y Gate

    Parameters
    ----------
    target:
        int or list of int
    control
        int or list of int
    power
        numeric type (fixed exponent) or hashable type (parametrized exponent)
    angle
        similar to power, but will be interpreted as
        .. math::
           U(\\text{angle})=e^{-i\\frac{angle}{2} (1-Y)}
        the default is angle=pi
        .. math::
           U(\\pi) = Y
        If angle and power are given both, tequila will combine them

    Returns
    -------
    QCircuit object

    """
    generator = lambda q: paulis.Y(q) - paulis.I(q)
    return _initialize_power_gate(name="Y", power=power, angle=angle, target=target, control=control,
                                  generator=generator)


def Z(target: typing.Union[list, int], control: typing.Union[list, int] = None, power=None, angle=None, *args, **kwargs) -> QCircuit:
    """
    Notes
    ----------
    Pauli Z Gate

    Parameters
    ----------
    target
        int or list of int
    control
        int or list of int
    power
        numeric type (fixed exponent) or hashable type (parametrized exponent)
    angle
        similar to power, but will be interpreted as
        .. math::
           U(\\text{angle})=e^{-i\\frac{angle}{2} (1-Z)}
        the default is angle=pi
        .. math::
           U(\\pi) = Z
        If angle and power are given both, tequila will combine them

    Returns
    -------
    QCircuit object

    """
    generator = lambda q: paulis.Z(q) - paulis.I(q)
    return _initialize_power_gate(name="Z", power=power, angle=angle, target=target, control=control,
                                  generator=generator, *args, **kwargs)


def ExpPauli(paulistring: typing.Union[PauliString, str], angle, control: typing.Union[list, int] = None, *args, **kwargs):
    """Exponentiated Pauligate:
    
    ExpPauli(PauliString, angle) = exp(-i* angle/2* PauliString)

    Parameters
    ----------
    paulistring :
        given as PauliString structure or as string or dict or list
        if given as string: Format should be like X(0)Y(3)Z(2)
        if given as list: Format should be like [(0,'X'),(3,'Y'),(2,'Z')]
        if given as dict: Format should be like { 0:'X', 3:'Y', 2:'Z' }
    angle :
        the angle (will be multiplied by paulistring coefficient if there is one)
    control :
        control qubits
    paulistring: typing.Union[PauliString :
        
    str] :
        
    control: typing.Union[list :
        
    int] :
         (Default value = None)

    Returns
    -------
    type
        Gate wrapped in circuit

    """

    if isinstance(paulistring, str):
        ps = PauliString.from_string(string=paulistring)
    elif isinstance(paulistring, list):
        ps = PauliString.from_openfermion(key=list)
    elif isinstance(paulistring, dict):
        ps = PauliString(data=paulistring)
    else:
        ps = paulistring

    # Failsave: If the paulistring contains just one pauli matrix
    # it is better to initialize a rotational gate due to strange conventions in some simulators
    if len(ps.items()) == 1:
        target, axis = tuple(ps.items())[0]
        return QCircuit.wrap_gate(
            impl.RotationGateImpl(axis=axis, target=target, angle=ps.coeff * assign_variable(angle), control=control, *args, **kwargs))
    else:
        return QCircuit.wrap_gate(impl.ExponentialPauliGateImpl(paulistring=ps, angle=angle, control=control, *args, **kwargs))


def Rp(paulistring: typing.Union[PauliString, str], angle, control: typing.Union[list, int] = None, *args, **kwargs):
    """
    Same as ExpPauli
    """
    return ExpPauli(paulistring=paulistring, angle=angle, control=control, *args, **kwargs)


def GeneralizedRotation(angle: typing.Union[typing.List[typing.Hashable], typing.List[numbers.Real]],
                        generator: QubitHamiltonian,
                        control: typing.Union[list, int] = None,
                        eigenvalues_magnitude: float = 0.5,
                        steps: int = 1, assume_real=False) -> QCircuit:
    """

    Notes
    --------
    
    A gates which is shift-rule differentiable
     - its generator only has two distinguishable eigenvalues
     - it is then differentiable by the shift rule
     - eigenvalues_magnitude needs to be given upon initialization (this is "r" from Schuld et. al. and the default is r=1/2)
     - the generator will not (!) be verified to fullfill the properties
     Compiling will be done in analogy to a trotterized gate with steps=1 as default

    The gate will act in the same way as rotations and exppauli gates

    .. math::
        U_{G}(\\text{angle}) = e^{-i\\frac{\\text{angle}}{2} G}
    
    Parameters
    ----------
    angle
        numeric type or hashable symbol or tequila objective
    generator
        tequila QubitHamiltonian or any other structure with paulistrings
    control
        list of control qubits
    eigenvalues_magnitude
        magnitude of eigenvalues, in most papers referred to as "r" (default 0.5)
    steps
        possible Trotterization steps (default 1)

    Returns
    -------
    The gate wrapped in a circuit
    """

    return QCircuit.wrap_gate(
        impl.GeneralizedRotationImpl(angle=assign_variable(angle), generator=generator, control=control,
                                eigenvalues_magnitude=eigenvalues_magnitude, steps=steps, assume_real=assume_real))




def Trotterized(generator: QubitHamiltonian = None,
                steps: int = 1,
                angle: typing.Union[typing.Hashable, numbers.Real, Variable] = None,
                control: typing.Union[list, int] = None,
                randomize=False,
                *args, **kwargs) -> QCircuit:
    """

    Parameters
    ----------
    generator :
        generator of the gate U = e^{-i\frac{angle}{2} G }
    angles :
        coefficients for each generator
    steps :
        trotter steps
    control :
        control qubits
    generators: QubitHamiltonian :
        The generator of the gate
    steps: int :
        Trotter Steps
    angle: typing.Hashable :
        A symbol that will be converted to a tq.Variable
    numbers.Real :
        A fixed real number
    Variable :
        A tequila Variable
    control: control qubits
    Returns
    -------
    QCircuit

    """

    #  downward compatibility
    if "generators" in kwargs:
        if generator is None:
            if len(kwargs["generators"]) > 1:
                if "angles" not in kwargs:
                    angles = [angle]*len(kwargs["generators"])
                else:
                    angles = kwargs["angles"]
                result = QCircuit()
                for angle,g in zip(angles,kwargs["generators"]):
                    result += Trotterized(generator=g, angle=angle, steps=steps, control=control, randomize=randomize)
                    return result
            else:
                generator = kwargs["generators"][0]
        else:
            raise Exception("Trotterized: You gave generators={} and generator={}".format(generator, kwargs["generators"]))

    if "angles" in kwargs:
        if angle is None:
            if len(kwargs["angles"]) > 1:
                raise Exception("multiple angles given, but only one generator")
            angle = kwargs["angles"][0]
        else:
            raise Exception("Trotterized: You gave angles={} and angle={}".format(angle, kwargs["angles"]))

    angle = assign_variable(angle)

    return QCircuit.wrap_gate(impl.TrotterizedGateImpl(generator=generator, angle=angle, steps=steps, control=control, randomize=randomize, **kwargs))


def SWAP(first: int, second: int, control: typing.Union[int, list] = None, power: float = None, *args,
         **kwargs) -> QCircuit:
    """
    Notes
    ----------
    SWAP gate, order of targets does not matter

    Parameters
    ----------
    first: int
        target qubit
    second: int
        target qubit
    control
        int or list of ints
    power
        numeric type (fixed exponent) or hashable type (parametrized exponent)

    Returns
    -------
    QCircuit

    """

    target = [first, second]
    generator = 0.5 * (paulis.X(target) + paulis.Y(target) + paulis.Z(target) - paulis.I(target))
    if power is None or power in [1, 1.0]:
        return QGate(name="SWAP", target=target, control=control, generator=generator)
    else:
        return GeneralizedRotation(angle=power * np.pi, control=control, generator=generator,
                                   eigenvalues_magnitude=0.25)


"""
Convenience Initialization Routines for controlled gates following the patern: Gate(control_qubit, target_qubit, possible_parameter)
All can be initialized as well with the standard operations above
"""


def CNOT(control: int, target: int) -> QCircuit:
    """
    Convenience CNOT initialization

    Parameters
    ----------
    control: int
        control qubit
    target: int
        target qubit

    Returns
    -------
    QCircuit object
    """
    return X(target=target, control=control)


def Toffoli(first: int, second: int, target: int) -> QCircuit:
    """
    Convenience Toffoli initialization

    Parameters
    ----------
    first: int
        first control qubit
    second: int
        second control qubit
    target: int
        target qubit

    Returns
    -------
    QCircuit object

    """
    return X(target=target, control=[first, second])


def CX(control: int, target: int) -> QCircuit:
    """
    Convenience initialization CX (CNOT)

    Parameters
    ----------
    control: int
        control qubit
    target: int
        target qubit

    Returns
    -------
    QCircuit object
    """
    return X(target=target, control=control)


def CY(control: int, target: int) -> QCircuit:
    """
    Convenience initialization CY (controlled Pauli Y)

    Parameters
    ----------
    control: int
        control qubit
    target: int
        target qubit

    Returns
    -------
    QCircuit object
    """
    return Y(target=target, control=control)


def CZ(control: int, target: int) -> QCircuit:
    """
    Convenience initialization CZ (controlled Pauli Z)

    Parameters
    ----------
    control: int
        control qubit
    target: int
        target qubit

    Returns
    -------
    QCircuit object
    """
    return Z(target=target, control=control)


def CRx(control: int, target: int, angle: float) -> QCircuit:
    """
    Convenience initialization CRx (controlled Pauli X Rotation)

    Parameters
    ----------
    control: int
        control qubit
    target: int
        target qubit
    angle:
        Hashable type (will be treated as Variable) or Numeric type (static angle)

    Returns
    -------
    QCircuit object
    """
    return Rx(target=target, control=control, angle=angle)


def CRy(control: int, target: int, angle: float) -> QCircuit:
    """
    Convenience initialization CRy (controlled Pauli Y Rotation)

    Parameters
    ----------
    control: int
        control qubit
    target: int
        target qubit
    angle:
        Hashable type (will be treated as Variable) or Numeric type (static angle)

    Returns
    -------
    QCircuit object
    """
    return Ry(target=target, control=control, angle=angle)


def CRz(control: int, target: int, angle: float) -> QCircuit:
    """
    Convenience initialization CRz (controlled Pauli Z Rotation)

    Parameters
    ----------
    control: int
        control qubit
    target: int
        target qubit
    angle:
        Hashable type (will be treated as Variable) or Numeric type (static angle)

    Returns
    -------
    QCircuit object
    """
    return Rz(target=target, control=control, angle=angle)


def U(theta, phi, lambd, target: typing.Union[list, int], control: typing.Union[list, int] = None) -> QCircuit:
    """
    Notes
    ----------
    Convenient gate, one of the abstract gates defined by OpenQASM.

    .. math::
        U(\\theta, \\phi, \\lambda) = R_z(\\phi)R_x(-\\pi/2)R_z(\\theta)R_x(\\pi/2)R_z(\\lambda)
        U(\\theta, \\phi, \\lambda) = \\begin{pmatrix}
                                            e^{-i \\frac{\\phi}{2}} & 0 \\\\
                                            0 & e^{i \\frac{\\phi}{2}}
                                        \\end{pmatrix}
                                        \\begin{pmatrix}
                                            \\cos{-\\frac{\\pi}{4}} & -i \\sin{-\\frac{\\pi}{4}} \\\\
                                            -i \\sin{-\\frac{\\pi}{4}} & \\cos{-\\frac{\\pi}{4}}
                                        \\end{pmatrix}
                                        \\begin{pmatrix}
                                            e^{-i \\frac{\\theta}{2}} & 0 \\\\
                                            0 & e^{i \\frac{\\theta}{2}}
                                        \\end{pmatrix}
                                        \\begin{pmatrix}
                                            \\cos{\\frac{\\pi}{4}} & -i \\sin{\\frac{\\pi}{4}} \\\\
                                            -i \\sin{\\frac{\\pi}{4}} & \\cos{\\frac{\\pi}{4}}
                                        \\end{pmatrix}
                                        \\begin{pmatrix}
                                            e^{-i \\frac{\\lambda}{2}} & 0 \\\\
                                            0 & e^{i \\frac{\\lambda}{2}}
                                        \\end{pmatrix}

        U(\\theta, \\phi, \\lambda) = \\begin{pmatrix}
                                        \\cos{\\frac{\\theta}{2}} &
                                        -e^{i \\lambda} \\sin{\\frac{\\theta}{2}} \\\\
                                        e^{i \\phi} \\sin{\\frac{\\theta}{2}} &
                                        e^{i (\\phi+\\lambda)} \\cos{\\frac{\\theta}{2}}
                                      \\end{pmatrix}

    Parameters
    ----------
    theta
        first parameter angle
    phi
        second parameter angle
    lamnd
        third parameter angle
    target
        int or list of int
    control
        int or list of int

    Returns
    -------
    QCircuit object
    """

    theta = assign_variable(theta)
    phi = assign_variable(phi)
    lambd = assign_variable(lambd)
    pi_half = assign_variable(np.pi / 2)

    return Rz(angle=lambd, target=target, control=control) + \
           Rx(angle=pi_half, target=target, control=control) + \
           Rz(angle=theta, target=target, control=control) + \
           Rx(angle=-pi_half, target=target, control=control) + \
           Rz(angle=phi, target=target, control=control)


def u1(lambd, target: typing.Union[list, int], control: typing.Union[list, int] = None) -> QCircuit:
    """
    Notes
    ----------
    Convenient gate, one of the abstract gates defined by Quantum Experience Standard Header.
    Changes the phase of a carrier without applying any pulses.

    .. math::
        from OpenQASM 2.0 specification:
            u1(\\lambda) \\sim U(0, 0, \\lambda) = R_z(\\lambda) = e^{-i\\frac{\\lambda}{2} \\sigma_{z}}
        also is equal to:
            u1(\\lambda) = \\begin{pmatrix} 1 & 0 \\\\ 0 & e^{i\\lambda} \\end{pmatrix}
        which is the Tequila Phase gate:
            u1(\\lambda) = Phase(\\lambda)

    Parameters
    ----------
    lambd
        parameter angle
    target
        int or list of int
    control
        int or list of int

    Returns
    -------
    QCircuit object
    """

    return Phase(phi=lambd, target=target, control=control)


def u2(phi, lambd, target: typing.Union[list, int], control: typing.Union[list, int] = None) -> QCircuit:
    """
    Notes
    ----------
    Convenient gate, one of the abstract gates defined by Quantum Experience Standard Header.
    Uses a single \\pi/2-pulse.

    .. math::
        u2(\\phi, \\lambda) = U(\\pi/2, \\phi, \\lambda) = R_z(\\phi + \\pi/2)R_x(\\pi/2)R_z(\\lambda - \\pi/2)

        u2(\\phi, \\lambda) = \\frac{1}{\\sqrt{2}}
                              \\begin{pmatrix}
                                    1          & -e^{i\\lambda} \\\\
                                    e^{i\\phi} & e^{i(\\phi+\\lambda)}
                              \\end{pmatrix}

    Parameters
    ----------
    phi
        first parameter angle
    lambd
        second parameter angle
    target
        int or list of int
    control
        int or list of int

    Returns
    -------
    QCircuit object
    """

    return U(theta=np.pi / 2, phi=phi, lambd=lambd, target=target, control=control)


def u3(theta, phi, lambd, target: typing.Union[list, int], control: typing.Union[list, int] = None) -> QCircuit:
    """
    Notes
    ----------
    Convenient gate, one of the abstract gates defined by Quantum Experience Standard Header
    The most general single-qubit gate.
    Uses a pair of \\pi/2-pulses.

    .. math::
        u3(\\theta, \\phi, \\lambda) = U(\\theta, \\phi, \\lambda)
                                     = \\begin{pmatrix}
                                            \\cos{\\frac{\\5theta}{2}} &
                                            -e^{i \\lambda} \\sin{\\frac{\\theta}{2}} \\\\
                                            e^{i \\phi} \\sin{\\frac{\\theta}{2}} &
                                            e^{i (\\phi+\\lambda)} \\cos{\\frac{\\theta}{2}}
                                       \\end{pmatrix}

    Parameters
    ----------
    theta
        first parameter angle
    phi
        second parameter angle
    lambd
        third parameter angle
    target
        int or list of int
    control
        int or list of int

    Returns
    -------
    QCircuit object
    """

    return U(theta=theta, phi=phi, lambd=lambd, target=target, control=control)


def QubitExcitation(angle: typing.Union[numbers.Real, Variable, typing.Hashable], target: typing.List, control=None,
                    assume_real: bool = False, compile_options="optimized"):
    """
    A Qubit Excitation, as described under "qubit perspective" in https://doi.org/10.1039/D0SC06627C
    For the Fermionic operators under corresponding Qubit encodings: Use the chemistry interface
    Parameters
    ----------
    angle:
        the angle of the excitation unitary
    target:
        even number of qubit indices interpreted as [0,1,2,3....] = [(0,1), (2,3), ...]
        i.e. as qubit excitations from 0 to 1, 2 to 3, etc
    control:
        possible control qubits
    assume_real:
        assume the wavefunction on which this acts is always real (cheaper gradients: see https://doi.org/10.1039/D0SC06627C)

    Returns
    -------
        QubitExcitation gate wrapped into a tequila circuit

    """
    try:
        assert len(target) % 2 == 0
    except:
        raise Exception("QubitExcitation: Needs an even number of targets")

    return QCircuit.wrap_gate(QubitExcitationImpl(angle=angle, target=target, assume_real=assume_real, compile_options=compile_options))


"""
Helper Functions
"""

def _initialize_power_gate(name: str, target: typing.Union[list, int], generator,
                           control: typing.Union[list, int] = None, power=None, angle=None, *args, **kwargs) -> QCircuit:
    target = list_assignment(target)

    # allow angle instead of power in initialization for more consistency
    # if angle is given we just convert it
    if angle is not None:
        angle = assign_variable(angle)
        if power is not None:
            power = power * angle / np.pi
        else:
            power = angle / np.pi

    if power is None or power in [1, 1.0]:
        gates = [impl.QGateImpl(name=name, target=q, control=control, generator=generator(q)) for q in target]
    else:
        gates = [impl.PowerGateImpl(name=name, power=power, target=q, control=control, generator=generator(q), *args, **kwargs) for q in
                 target]

    return QCircuit.wrap_gate(gates)


def RotationGate(axis: int, angle: typing.Union[typing.Hashable, numbers.Number], target: typing.Union[list, int], control: typing.Union[list, int] = None, assume_real=False):
    """
    Notes
    ----------
    Initialize an abstract rotation gate of the form

    .. math::
        R_{\\text{axis}}(\\text{angle}) = e^{-i\\frac{\\text{angle}}{2} \\sigma_{\\text{axis}}}


    Parameters
    ----------
    axis
        integer 1 for x, 2 for y, 3 for z
    angle
        Hashable type (will be treated as Variable) or Numeric type (static angle)
    target
        integer or list of integers
    control
        integer or list of integers
    assume_real
        enable improved gradient compilation for controlled gates (wavefunction needs to be real)

    Returns
    -------
    QCircuit object with this RotationGate
    """
    target = list_assignment(target)
    gates = [impl.RotationGateImpl(axis=axis, angle=angle, target=q, control=control, assume_real=assume_real) for q in target]

    return QCircuit.wrap_gate(gates)


def PowerGate(name: str, target: typing.Union[list, int], power: float = None, control: typing.Union[list, int] = None, generator: QubitHamiltonian = None, *args, **kwargs):
    """
    Initialize a (potentially parametrized) gate which is supported on the backend

    Parameters
    ----------
    name: str
        name of the gate on the backend (usually, H, X, Y, Z)
    target
        int or list of int
    power
        numeric type (fixed exponent) or hashable type (parametrized exponent)
        will be interpreted as
    angle
        similar to power, but will be interpreted as
        .. math::
           U=e^{-i\\frac{angle}{2} generator}
    control
        int or list of int

    Returns
    -------

    """
    return QCircuit.wrap_gate(
        impl.PowerGateImpl(name=name, power=power, target=target, control=control, generator=generator, *args, **kwargs))


def QGate(name, target: typing.Union[list, int], control: typing.Union[list, int] = None,
          generator: QubitHamiltonian = None):
    return QCircuit.wrap_gate(impl.QGateImpl(name=name, target=target, control=control, generator=generator))

"""
Implementation of specific gates
Not put into _gates_impl.py for convenience
Those gate types will not be recognized by the compiler
and should all implement a compile function that 
returns a QCircuit of primitive tq gates
"""

class QubitExcitationImpl(impl.DifferentiableGateImpl):
    @staticmethod
    def extract_targets(generator):
        targets = []
        for ps in generator.paulistrings:
            targets += [k for k in ps.keys()]
        return tuple(set(targets))

    @property
    def steps(self):
        return 1

    def __init__(self, angle, target=None, generator=None, p0=None, assume_real=True, control=None, compile_options=None):
        angle = assign_variable(angle)

        if generator is None:
            assert target is not None
            assert p0 is None
            generator = paulis.I()
            p0a = paulis.I()
            p0b = paulis.I()

            for i in range(len(target) // 2):
                generator *= paulis.Sp(target[2 * i]) * paulis.Sm(target[2 * i + 1])
                p0a *= paulis.Qp(target[2 * i]) * paulis.Qm(target[2 * i + 1])
                p0b *= paulis.Qm(target[2 * i]) * paulis.Qp(target[2 * i + 1])
            generator = (1.0j * (generator - generator.dagger())).simplify()
            p0 = paulis.I() - p0a - p0b
        else:
            assert generator is not None
            assert p0 is not None

        super().__init__(name="QubitExcitation", parameter=angle, target=self.extract_targets(generator), control=control)
        self.generator = generator
        if control is not None:
            # augment p0 for control qubits
            # Qp = 1/2(1+Z) = |0><0|
            p0 = p0*paulis.Qp(control)
        self.p0 = p0
        self.assume_real = assume_real
        if compile_options is None:
            self.compile_options = "optimize"
        else:
            self.compile_options = compile_options

    def map_qubits(self, qubit_map: dict):
        mapped_generator = self.generator.map_qubits(qubit_map=qubit_map)
        mapped_p0 = self.p0.map_qubits(qubit_map=qubit_map)
        mapped_control = self.control
        if mapped_control is not None:
            mapped_control=tuple([qubit_map[i] for i in self.control])
        return type(self)(angle=self.parameter, generator=mapped_generator, p0=mapped_p0, assume_real=self.assume_real, control=mapped_control)


    def compile(self):
        # optimized compiling for single and double qubit excitaitons following arxiv:2005.14475
        # Alternative representation in arxiv:2104.05695 (not implemented -> could be added and controlled with optional compile keywords)
        if self.compile == "optimize" and len(self.target) == 2:
            p,q = self.target
            U0 = X(target=p)
            U0 += X(target=p, control=q)
            U0 += X(target=p)
            U1 = Ry(angle=self.parameter, target=q, control=p)
            return U0 + U1 + U0
        elif self.compile == "optimize" and len(self.target) == 4:
            p,q,r,s = self.target
            U0 = X(target=q, control=p)
            U0 += X(target=s, control=r)
            U0 += X(target=r, control=p)
            U0 += X(target=q)
            U0 += X(target=s)
            U1 = Ry(angle=-self.parameter, target=p, control=[q,r,s])
            return U0 + U1 + U0.dagger()
        else:
            return Trotterized(angle=self.parameter, generator=self.generator, steps=1)

    def shifted_gates(self):
        r = 0.25
        s = 0.5*np.pi

        Up1 = copy.deepcopy(self)
        Up1._parameter=self.parameter+s
        Up1 = QCircuit.wrap_gate(Up1)
        Up2 = GeneralizedRotation(angle=s, generator=self.p0, eigenvalues_magnitude=r) # controls are in p0
        Um1 = copy.deepcopy(self)
        Um1._parameter=self.parameter-s
        Um1 = QCircuit.wrap_gate(Um1)
        Um2 = GeneralizedRotation(angle=-s, generator=self.p0, eigenvalues_magnitude=r) # controls are in p0

        if not self.assume_real:
            # 4-point shift rule of arxiv:2104.05695 saves gates
            return [(r, Up1 + Up2), (-r, Um1 + Um2), (r, Up1 + Um2), (-r, Um1 + Up2)]
        else:
            return [(2.0 * r, Up1 +  Up2), (-2.0 * r, Um1 + Um2)]

