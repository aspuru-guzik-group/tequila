from tequila import TequilaException
from tequila.circuit.circuit import QCircuit
from tequila.circuit.gates import Rx, Ry, H, X, Rz, ExpPauli, CNOT, Phase, T, Z, Y, S, CX
from tequila.circuit._gates_impl import RotationGateImpl, PhaseGateImpl, QGateImpl, \
    ExponentialPauliGateImpl, TrotterizedGateImpl, PowerGateImpl
from tequila.utils import to_float
from tequila.objective.objective import Variable
from tequila.objective.objective import Objective, VectorObjective
from tequila.objective.objective import ExpectationValueImpl
from tequila.autograd_imports import numpy as jnp
import numpy
from numpy import pi as pi

import copy, typing
import time


class TequilaCompilerException(TequilaException):
    pass


class Compiler:
    """
    an object that performs abstract compilation of QCircuits and Objectives.

    Note
    ----
        see init for attributes, since all are specified there

    Methods
    -------
    compile_objective
        perform compilation on an entire objective
    compile_objective_argument
        perform compilation on a single arg of objective
    compile_circuit:
        perform compilation on a circuit.
    """

    def __init__(self,
                 multitarget=False,
                 multicontrol=False,
                 trotterized=False,
                 generalized_rotation=False,
                 exponential_pauli=False,
                 controlled_exponential_pauli=False,
                 hadamard_power=False,
                 controlled_power=False,
                 power=False,
                 toffoli=False,
                 controlled_phase=False,
                 phase=False,
                 phase_to_z=False,
                 controlled_rotation=False,
                 swap=False,
                 cc_max=False,
                 gradient_mode=False,
                 ry_gate=False,
                 y_gate=False,
                 ch_gate=False,
                 hadamard=False
                 ):

        """
        all parameters are booleans.
        Parameters
        ----------
        multitarget:
            whether or not to split multitarget gates into single target (if gate isn't inherently multitarget)
        multicontrol:
            whether or not to split gates into single controlled gates.
        trotterized:
            whether or not to break down TrotterizedGateImpl into other types
        generalized_rotation:
            whether or not to break down GeneralizedRotationGateImpl into other types
        exponential_pauli:
            whether or not to break down ExponentialPauliGateImpl into other types
        controlled_exponential_pauli
            whether or not to break down controlled exponential pauli gates.
        hadamard_power:
            whether or not to break down Hadamard gates, raised to a power, into other rotation gates.
        controlled_power:
            whether or not to break down controlled power gates into CNOT and other gates.
        power:
            whether or not to break down parametrized power gates into rotation gates
        toffoli:
            whether or not to break down the toffoli gate into CNOTs and other single qubit gates.
        controlled_phase:
            whether or not to break down controlled phase gates into CNOTs and phase gates.
        phase:
            whether to replace phase gates
        phase_to_z:
            specifically, whether to replace phase gates with the z gate
        controlled_rotation:
            whether or not to break down controlled rotation gates into CNot and single qubit gates
        swap:
            whether or not to break down swap gates into CNOT gates.
        cc_max:
            whether or not to break down all controlled gates with 2 or more controls.
        ry_gate:
            whether or not to break down all rotational y gates
        y_gate:
            whether or not to break down all y gates
        ch_gate:
            whether or not to break down all controlled-H gates
        """
        self.multitarget = multitarget
        self.multicontrol = multicontrol
        self.generalized_rotation = generalized_rotation
        self.trotterized = trotterized
        self.exponential_pauli = exponential_pauli
        self.controlled_exponential_pauli = controlled_exponential_pauli
        self.hadamard_power = hadamard_power
        self.hadamard = hadamard
        self.controlled_power = controlled_power
        self.power = power
        self.toffoli = toffoli
        self.controlled_phase = controlled_phase
        self.phase = phase
        self.phase_to_z = phase_to_z
        self.controlled_rotation = controlled_rotation
        self.swap = swap
        self.cc_max = cc_max
        self.gradient_mode = gradient_mode
        self.ry_gate = ry_gate
        self.y_gate = y_gate
        self.ch_gate = ch_gate

    def __call__(self, objective: typing.Union[Objective, QCircuit, ExpectationValueImpl], variables=None, *args,
                 **kwargs):

        """
        Perform compilation
        Parameters
        ----------
        objective:
            the object (not necessarily an objective) to compile.
        variables: optional:
            Todo: Jakob, what is this for?
        args
        kwargs

        Returns
        -------
        a compiled version of objective
        """

        if isinstance(objective, Objective) or hasattr(objective, "args"):
            result = self.compile_objective(objective=objective, variables=variables, *args, **kwargs)
        elif isinstance(objective, QCircuit) or hasattr(objective, "gates"):
            result = self.compile_circuit(abstract_circuit=objective, variables=variables, *args, **kwargs)
        elif isinstance(objective, ExpectationValueImpl) or hasattr(objective, "U"):
            result = self.compile_objective_argument(arg=objective, variables=variables, *args, **kwargs)
        else:
            raise TequilaCompilerException("Tequila compiler can't process type {}".format(type(objective)))

        return result

    def compile_objective(self, objective, *args, **kwargs):
        """
        Compile an objective.

        Parameters
        ----------
        objective: Objective:
            the objective.
        args
        kwargs
        Returns
        -------
        the objective, compiled
        """

        argsets=objective.argsets
        compiled_sets=[]
        for argset in argsets:
            compiled_args = []
            already_processed = {}
            for arg in argset:
                if isinstance(arg, ExpectationValueImpl) or (hasattr(arg, "U") and hasattr(arg, "H")):
                    if arg in already_processed:
                        compiled_args.append(already_processed[arg])
                    else:
                        compiled = self.compile_objective_argument(arg, *args, **kwargs)
                        compiled_args.append(compiled)
                        already_processed[arg] = compiled
                else:
                    # nothing to process for non-expectation-value types, but acts as sanity check
                    compiled_args.append(self.compile_objective_argument(arg, *args, **kwargs))
            compiled_sets.append(compiled_args)
        if isinstance(objective,Objective):
            return type(objective)(args=compiled_sets[0],transformation=objective.transformation)
        if isinstance(objective, VectorObjective):
            return type(objective)(argsets=compiled_sets, transformations=objective.transformations)


    def compile_objective_argument(self, arg, *args, **kwargs):
        """
        Compile an argument of an objective.

        Parameters
        ----------
        arg:
            the term to compile
        args
        kwargs

        Returns
        -------
        the arg, compiled
        """


        if isinstance(arg, ExpectationValueImpl) or (hasattr(arg, "U") and hasattr(arg, "H")):
            return ExpectationValueImpl(H=arg.H,
                                        U=self.compile_circuit(abstract_circuit=arg.U, *args,
                                                               **kwargs))
        elif hasattr(arg, "abstract_expectationvalue"):
            E = arg.abstract_expectationvalue
            E._U = self.compile_circuit(abstract_circuit=E.U, *args, **kwargs)
            return type(arg)(E, **arg._input_args)
        elif isinstance(arg, Variable) or hasattr(arg, "name"):
            return arg
        else:
            raise TequilaCompilerException(
                "Unknown argument type for objectives: {arg} or type {type}".format(arg=arg, type=type(arg)))

    def compile_circuit(self, abstract_circuit: QCircuit, variables=None, *args, **kwargs) -> QCircuit:
        """
        compile a circuit.
        Parameters
        ----------
        abstract_circuit: QCircuit
            the circuit to compile.
        variables:
            (Default value = None):
            list of the variables whose gates, specifically, must compile.
            Used to prevent excess compilation in gates whose parameters are fixed.
            Default: compile every single gate.
        args
        kwargs

        Returns
        -------
            QCircuit; a compiled circuit.
        """

        n_qubits = abstract_circuit.n_qubits
        compiled = QCircuit(abstract_circuit.gates)

        if variables is None:
            # check & compile all gates
            gatelist = enumerate(abstract_circuit.gates)
        else:
            # check & compile only gates which depend on variables
            gatelist = []
            for variable in variables:
                gatelist += abstract_circuit._parameter_map[variable]

        compiled_gates = []

        for idx, gate in gatelist:

            cg = gate
            controlled = gate.is_controlled()

            if self.gradient_mode and (hasattr(cg, "eigenvalues_magnitude") or hasattr(cg, "shifted_gates")):
                compiled_gates.append((idx, QCircuit.wrap_gate(cg)))
                continue
            else:
                if hasattr(cg, "compile"):
                    cg = QCircuit.wrap_gate(cg.compile())
                    for g in cg.gates:
                        if g.is_controlled():
                            controlled = True


            # order matters
            # first the real multi-target gates
            if controlled or self.trotterized:
                cg = compile_trotterized_gate(gate=cg)
            if controlled or self.generalized_rotation:
                cg = compile_generalized_rotation_gate(gate=cg)
            if controlled or self.exponential_pauli:
                cg = compile_exponential_pauli_gate(gate=cg)
            if self.swap:
                cg = compile_swap(gate=cg)
            if self.phase_to_z:
                cg = compile_phase_to_z(gate=cg)
            if self.power:
                cg = compile_power_gate(gate=cg)
            if self.phase:
                cg = compile_phase(gate=cg)
            if self.ch_gate:
                cg = compile_ch(gate=cg)
            if self.y_gate:
                cg = compile_y(gate=cg)
            if self.ry_gate:
                cg = compile_ry(gate=cg, controlled_rotation=self.controlled_rotation)
            if controlled:
                if self.cc_max or self.multicontrol:
                    cg = compile_to_single_control(gate=cg)
                if self.controlled_exponential_pauli:
                    cg = compile_exponential_pauli_gate(gate=cg)
                if self.controlled_power:
                    cg = compile_controlled_power(gate=cg)
                if self.controlled_phase:
                    cg = compile_controlled_phase(gate=cg)
                    if self.phase:
                        cg = compile_phase(gate=cg)
                if self.toffoli:
                    cg = compile_toffoli(gate=cg)
                    if self.phase:
                        cg = compile_phase(gate=cg)
                if self.controlled_rotation:
                    cg = compile_controlled_rotation(gate=cg)

            compiled_gates.append((idx, cg))

        if len(compiled_gates) == 0:
            return abstract_circuit
        else:
            pos, cgs = zip(*compiled_gates)
            compiled = abstract_circuit.replace_gates(positions=pos, circuits=cgs)

            return compiled


def compiler(f):
    """
    Decorator for compile functions.

    Make them applicable for single gates as well as for whole circuits
    Note that all arguments need to be passed as keyword arguments
    """

    def wrapper(gate, **kwargs):
        if hasattr(gate, "gates"):
            result = QCircuit()
            for g in gate.gates:
                result += f(gate=g, **kwargs)
            return result

        elif hasattr(gate, 'U'):
            cU = QCircuit()
            for g in gate.U.gates:
                cU += f(gate=g, **kwargs)
            return type(gate)(U=cU, H=gate.H)
        elif hasattr(gate, 'transformations'):
            outer=[]
            for args in gate.argsets:
                compiled = []
                for E in args:
                    if hasattr(E, 'name'):
                        compiled.append(E)
                    else:
                        cU = QCircuit()
                        for g in E.U.gates:
                            cU += f(gate=g, **kwargs)
                        compiled.append(type(E)(U=cU, H=E.H))
                outer.append(compiled)
            if isinstance(gate, Objective):
                return type(gate)(args=outer[0], transformation=gate._transformation)
            if isinstance(gate, VectorObjective):
                return type(gate)(argsets=outer, transformations=gate._transformations)
        else:
            return f(gate=gate, **kwargs)

    return wrapper


def change_basis(target, axis=None, name=None, daggered=False):
    """
    helper function; returns circuit that performs change of basis.
    Parameters
    ----------
    target:
        the qubit having its basis changed
    axis:
        The axis of rotation to shift into.
    daggered: bool:
        adjusts the sign of the gate if axis = 1, I.E, change of basis about Y axis.

    Returns
    -------
    QCircuit that performs change of basis on target qubit onto desired axis

    """
    if axis is None and name is None:
        raise TequilaException('axis or name must be given.')

    if name:
        name = name.lower()
        if name in ['h', 'hadamard'] and daggered:
            return Ry(angle=numpy.pi / 4, target=target)
        elif name in ['h', 'hadamard']:
            return Ry(angle=-numpy.pi / 4, target=target)
        else:
            name_to_axis = {'rx': 0, 'ry': 1, 'rz': 2}
            axis = name_to_axis.get(name, name)

    if isinstance(axis, str):
        axis = RotationGateImpl.string_to_axis[axis.lower()]

    if axis == 0 and daggered:
        return Ry(angle=numpy.pi / 2, target=target)
    elif axis == 0:
        return Ry(angle=-numpy.pi / 2, target=target)
    elif axis == 1 and daggered:
        return Rx(angle=-numpy.pi / 2, target=target)
    elif axis == 1:
        return Rx(angle=numpy.pi / 2, target=target)
    else:
        return QCircuit()

@compiler
def compile_multitarget(gate, *args, **kwargs) -> QCircuit:
    """
    If a gate is 'trivially' multitarget, split it into single target gates.
    Parameters
    ----------
    gate:
        the gate in question

    Returns
    -------
    QCircuit, the result of compilation.
    """
    targets = gate.target

    # don't compile real multitarget gates
    if hasattr(gate, "generator") or hasattr(gate, "generators") or hasattr(gate, "paulistring"):
        return QCircuit.wrap_gate(gate)

    if isinstance(gate, ExponentialPauliGateImpl) or isinstance(gate, TrotterizedGateImpl):
        return QCircuit.wrap_gate(gate)

    if len(targets) == 1:
        return QCircuit.wrap_gate(gate)

    if gate.name.lower() in ["swap", "iswap"]:
        return QCircuit.wrap_gate(gate)

    result = QCircuit()
    for t in targets:
        gx = copy.deepcopy(gate)
        gx._target = (t,)
        result += gx

    return result


# return index of control qubits in Gray Code order.
def _pattern(n):
    if n == 1:
        return [0]
    pn = _pattern(n - 1)

    return pn + [n - 1] + pn


@compiler
def compile_controlled_rotation(gate: RotationGateImpl) -> QCircuit:
    """
    Recompilation of a controlled-rotation gate
    Basis change into Rz then recompilation of controled Rz, then change basis back
    :param gate: The rotational gate
    :return: set of gates wrapped in QCircuit class
    """

    if not gate.is_controlled():
        return QCircuit.wrap_gate(gate)

    if not isinstance(gate, RotationGateImpl):
        return QCircuit.wrap_gate(gate)

    if len(gate.target) > 1:
        return compile_controlled_rotation(gate=compile_multitarget(gate=gate))

    target = gate.target
    control = gate.control
    k = len(control)
    cind = _pattern(k) + [k - 1]

    result = QCircuit()
    result += change_basis(target=target, axis=gate._axis)
    coeff = - 1 / pow(2, k)
    for i, ci in enumerate(cind):
        coeff *= -1

        result += Rz(target=target, angle=coeff * gate.parameter)
        result += CNOT(control[ci], target)
    result += change_basis(target=target, axis=gate._axis, daggered=True)

    result.n_qubits = result.max_qubit() + 1
    return result


@compiler
def compile_to_single_control(gate) -> QCircuit:
    """
    break down a gate into a sequence with no more than single-controlled gates.
    Parameters
    ----------
    gate:
        the gate.

    Returns
    -------
        A QCircuit; the result of compilation.
    """
    if not gate.is_controlled:
        return QCircuit.wrap_gate(gate)
    cl = len(gate.control)
    target = gate.target
    control = gate.control
    if cl <= 1:
        return QCircuit.wrap_gate(gate)
    name = gate.name
    back = QCircuit()
    if name in ['X', 'x', 'Y', 'y', 'Z', 'z', 'H', 'h']:
        if isinstance(gate, PowerGateImpl):
            power = gate.parameter
        else:
            power = 1.0
        new = PowerGateImpl(name=name, power=power, target=target, control=control, generator=gate.make_generator())
        partial = compile_power_gate(gate=new)
        back += compile_to_single_control(gate=partial)
    elif isinstance(gate, RotationGateImpl):
        partial = compile_controlled_rotation(gate=gate)
        back += compile_to_single_control(gate=partial)
    elif isinstance(gate, PhaseGateImpl):
        partial = compile_controlled_phase(gate=gate)
        back += compile_to_single_control(gate=partial)
    else:
        print(gate)
        raise TequilaException('frankly, what the fuck is this gate?')
    return back


@compiler
def compile_toffoli(gate) -> QCircuit:
    """
    break down a toffoli gate into a sequence of CNOT and single qubit gates.
    Parameters
    ----------
    gate:
        the gate.

    Returns
    -------
        A QCircuit; the result of compilation.
    """

    if gate.name.lower != 'x':
        return QCircuit.wrap_gate(gate)
    control = gate.control
    c1 = control[1]
    c0 = control[0]
    target = gate.target
    result = QCircuit()
    result += H(target)
    result += CNOT(c1, target)
    result += T(target).dagger()
    result += CNOT(c0, target)
    result += T(target)
    result += CNOT(c1, target)
    result += T(target).dagger()
    result += CNOT(c0, target)
    result += T(c1)
    result += T(target)
    result += CNOT(c0, c1)
    result += H(target)
    result += T(c0)
    result += T(c1).dagger()
    result += CNOT(c0, c1)

    return (result)


@compiler
def compile_power_gate(gate) -> QCircuit:
    """
    break down power gates into the rotation gates.
    Parameters
    ----------
    gate:
        the gate.

    Returns
    -------
        A QCircuit; the result of compilation.
    """
    if not isinstance(gate, PowerGateImpl):
        return QCircuit.wrap_gate(gate)
    if not gate.is_controlled():
        return compile_power_base(gate=gate)

    return compile_controlled_power(gate=gate)


@compiler
def compile_power_base(gate):
    """
    Base case of compile_power_gate: convert a 1-qubit parametrized power gate into rotation gates.
    Parameters
    ----------
    gate:
        the gate.

    Returns
    -------
        A QCircuit; the result of compilation.
    """
    if not isinstance(gate, PowerGateImpl):
        return QCircuit.wrap_gate(gate)

    if gate.is_controlled():
        return QCircuit.wrap_gate(gate)

    power = gate.power
    if gate.name.lower() in ['h', 'hadamard']:
        ### off by global phase of Exp[ pi power /2]
        theta = power * numpy.pi

        result = QCircuit()
        result += Ry(angle=-numpy.pi / 4, target=gate.target)
        result += Rz(angle=theta, target=gate.target)
        result += Ry(angle=numpy.pi / 4, target=gate.target)
    elif gate.name == 'X':
        ### off by global phase of Exp[ pi power /2]
        '''
        if we wanted to do it formally we would use the following
        a=-numpy.pi/2
        b=numpy.pi/2
        theta = power*numpy.pi

        result = QCircuit()
        result+= Rz(angle=b,target=gate.target)
        result+= Ry(angle=theta,target=gate.target)
        result+= Rz(angle=a,target=gate.target)
        '''
        result = Rx(angle=power * numpy.pi, target=gate.target)
    elif gate.name == 'Y':
        ### off by global phase of Exp[ pi power /2]
        theta = power * numpy.pi

        result = QCircuit()
        result += Ry(angle=theta, target=gate.target)
    elif gate.name == 'Z':
        ### off by global phase of Exp[ pi power /2]
        a = 0
        b = power * numpy.pi
        theta = 0
        result = QCircuit()
        result += Rz(angle=b, target=gate.target)
    else:
        raise TequilaException('passed a gate with name ' + gate.name + ', which cannot be handled!')
    return result


@compiler
def compile_controlled_power(gate: PowerGateImpl) -> QCircuit:
    """
    Recompilation of a controlled-power gate
    Basis change into Z then recompilation of controled Z, then change basis back
    :param gate: The power gate
    :return: set of gates wrapped in QCircuit class
    """
    if not gate.is_controlled():
        return QCircuit.wrap_gate(gate)

    if not isinstance(gate, PowerGateImpl):
        return QCircuit.wrap_gate(gate)

    if len(gate.target) > 1:
        return compile_controlled_power(gate=compile_multitarget(gate=gate))

    power = gate.power
    target = gate.target
    control = gate.control

    result = QCircuit()
    result += Phase(target=control[0], control=control[1:], phi=power * pi / 2)
    result += change_basis(target=target, name=gate.name)
    result += Rz(target=target, control=control, angle=power * pi)
    result += change_basis(target=target, name=gate.name, daggered=True)

    result.n_qubits = result.max_qubit() + 1
    return result


@compiler
def compile_phase(gate) -> QCircuit:
    """
    Compile phase gates into Rz gates and cnots, if controlled
    Parameters
    ----------
    gate:
        the gate

    Returns
    -------
    QCircuit, the result of compilation.
    """
    if not isinstance(gate, PhaseGateImpl):
        return QCircuit.wrap_gate(gate)
    phase = gate.parameter
    result = QCircuit()
    if len(gate.control) == 0:
        return Rz(angle=phase, target=gate.target)

    result = compile_controlled_phase(gate)
    result = compile_phase(result)
    return result


@compiler
def compile_controlled_phase(gate) -> QCircuit:
    """
    Compile multi-controlled phase gates to 1q - phase gate and multi-controlled Rz gates.
    Parameters
    ----------
    gate:
        the gate.

    Returns
    -------
    QCircuit, the result of compilation.
    """
    if not isinstance(gate, PhaseGateImpl):
        return QCircuit.wrap_gate(gate)

    if len(gate.control) == 0:
        return QCircuit.wrap_gate(gate)

    phase = gate.parameter

    result = QCircuit()
    result += Phase(target=gate.control[0], control=gate.control[1:], phi=phase / 2)
    result += Rz(target=gate.target, control=gate.control, angle=phase)
    return compile_controlled_phase(result)


@compiler
def compile_phase_to_z(gate) -> QCircuit:
    """
    Compile phase gate to parametrized Z gate.
    Parameters
    ----------
    gate:
        the gate.

    Returns
    -------
    QCircuit, the result of compilation.

    """
    if not isinstance(gate, PhaseGateImpl):
        return QCircuit.wrap_gate(gate)
    phase = gate.parameter
    return Z(power=phase / pi, target=gate.target, control=gate.control)


@compiler
def compile_swap(gate) -> QCircuit:
    """
    Compile swap gates into CNOT.
    Parameters
    ----------
    gate:
        the gate.

    Returns
    -------
    QCircuit, the result of compilation.
    """
    if gate.name.lower() == "swap":
        if len(gate.target) != 2:
            raise TequilaCompilerException("SWAP gates needs two targets")
        power = 1
        if hasattr(gate, "power"):
            if power is None or power in [1, 1.0]:
                pass
            else:
                raise TequilaCompilerException("Parametrized SWAPs should be decomposed on top level! Something went wrong")

        c = []
        if gate.control is not None:
            c = gate.control
        return X(target=gate.target[0], control=[gate.target[1]]) \
               + X(target=gate.target[1], control=[gate.target[0]] + list(c), power=power) \
               + X(target=gate.target[0], control=[gate.target[1]])

    else:
        return QCircuit.wrap_gate(gate)


@compiler
def compile_exponential_pauli_gate(gate) -> QCircuit:
    """
    Returns the circuit: exp(i*angle*paulistring)
    primitively compiled into X,Y Basis Changes and CNOTs and Z Rotations
    :param paulistring: The paulistring in given as tuple of tuples (openfermion format)
    like e.g  ( (0, 'Y'), (1, 'X'), (5, 'Z') )
    :param angle: The angle which parametrizes the gate -> should be real
    :returns: the above mentioned circuit as abstract structure
    """

    if hasattr(gate, "paulistring"):

        angle = gate.paulistring.coeff * gate.parameter

        circuit = QCircuit()

        # the general circuit will look like:
        # series which changes the basis if necessary
        # series of CNOTS associated with basis changes
        # Rz gate parametrized on the angle
        # series of CNOT (inverted direction compared to before)
        # series which changes the basis back
        ubasis = QCircuit()
        ubasis_t = QCircuit()
        cnot_cascade = QCircuit()

        last_qubit = None
        previous_qubit = None
        for k, v in gate.paulistring.items():
            pauli = v
            qubit = [k]  # wrap in list for targets= ...

            # see if we need to change the basis
            axis = 2
            if pauli.upper() == "X":
                axis = 0
            elif pauli.upper() == "Y":
                axis = 1
            ubasis += change_basis(target=qubit, axis=axis)
            ubasis_t += change_basis(target=qubit, axis=axis, daggered=True)

            if previous_qubit is not None:
                cnot_cascade += X(target=qubit, control=previous_qubit)
            previous_qubit = qubit
            last_qubit = qubit

        reversed_cnot = cnot_cascade.dagger()

        # assemble the circuit
        circuit += ubasis
        circuit += cnot_cascade
        circuit += Rz(target=last_qubit, angle=angle, control=gate.control)
        circuit += reversed_cnot
        circuit += ubasis_t

        return circuit

    else:
        return QCircuit.wrap_gate(gate)


def do_compile_trotterized_gate(generator, steps, factor, randomize, control):

    assert (generator.is_hermitian())
    circuit = QCircuit()
    factor = factor / steps
    for index in range(steps):
        paulistrings = generator.paulistrings
        if randomize:
            numpy.random.shuffle(paulistrings)
        for ps in paulistrings:
            coeff = to_float(ps.coeff)
            if len(ps._data) == 0 and len(control) > 0:
                circuit += Phase(target=control[0], control=control[1:], phi=-factor * coeff / 2)
            elif len(ps._data) > 0:
                circuit += ExpPauli(paulistring=ps.naked(), angle=factor * coeff, control=control)
            else:
                # ignore global phases
                pass
    return circuit


@compiler
def compile_generalized_rotation_gate(gate, compile_exponential_pauli: bool = False):
    """
    Parameters
    ----------
    gate
    compile_exponential_pauli

    Returns
    -------

    """
    if gate.generator is None or gate.name.lower() in ['phase', 'rx', 'ry', 'rz']:
        return QCircuit.wrap_gate(gate)
    if not hasattr(gate, "eigenvalues_magnitude"):
        return QCircuit.wrap_gate(gate)

    steps = 1 if not hasattr(gate, "steps") else gate.steps

    return do_compile_trotterized_gate(generator=gate.generator, steps=steps, randomize=False,
                                       factor=gate.parameter, control=gate.control)


@compiler
def compile_trotterized_gate(gate, compile_exponential_pauli: bool = False):
    """
    Parameters
    ----------
    gate
    compile_exponential_pauli

    Returns
    -------

    """
    if not hasattr(gate, "steps") or hasattr(gate, "eigenvalues_magnitude"):
        return QCircuit.wrap_gate(gate)

    randomize=False
    if hasattr(gate, "randomize"):
        randomize=gate.randomize
    result = do_compile_trotterized_gate(generator=gate.generator, steps=gate.steps, factor=gate.parameter, randomize=randomize, control=gate.control)

    if compile_exponential_pauli:
        return compile_exponential_pauli_gate(result)
    else:
        return result


@compiler
def compile_ry(gate: RotationGateImpl, controlled_rotation: bool = False) -> QCircuit:
    """
    Compile Ry gates into Rx and Rz.
    Parameters
    ----------
    gate:
        the gate.
    controlled_rotation:
        determines if the decomposition of the controlled-Ry gate will be performed in compile_controlled_rotation,
        if not, decomposition will be performed here

    Returns
    -------
    QCircuit, the result of compilation.
    """
    if gate.name.lower() == "ry":

        if not (gate.is_controlled() and controlled_rotation):

            return Rz(target=gate.target, control=None, angle=-numpy.pi / 2) \
                   + Rx(target=gate.target, control=gate.control, angle=gate.parameter) \
                   + Rz(target=gate.target, control=None, angle=numpy.pi / 2)

    return QCircuit.wrap_gate(gate)


@compiler
def compile_y(gate) -> QCircuit:
    """
    Compile Y gates into X and Rz.
    Parameters
    ----------
    gate:
        the gate.

    Returns
    -------
    QCircuit, the result of compilation.
    """
    if gate.name.lower() == "y":

        return Rz(target=gate.target, control=None, angle=-numpy.pi / 2) \
               + X(target=gate.target, control=gate.control, power=gate.power if gate.is_parametrized() else None) \
               + Rz(target=gate.target, control=None, angle=numpy.pi / 2)

    else:
        return QCircuit.wrap_gate(gate)


@compiler
def compile_ch(gate: QGateImpl) -> QCircuit:
    """
    Compile CH gates into its equivalent:
        CH = Ry(0.25pi) CZ Ry(-0.25pi)
    Parameters
    ----------
    gate:
        the gate.

    Returns
    -------
    QCircuit, the result of compilation.
    """
    if gate.name.lower() == "h" and gate.is_controlled():

        return Ry(target=gate.target, control=None, angle=-numpy.pi / 4) \
               + Z(target=gate.target, control=gate.control, power=gate.power if gate.is_parametrized() else None) \
               + Ry(target=gate.target, control=None, angle=numpy.pi / 4)
    else:
        return QCircuit.wrap_gate(gate)
