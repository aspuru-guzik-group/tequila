import typing
import copy
import numbers
from abc import ABC
from tequila import TequilaException
from tequila.objective.objective import Variable, FixedVariable, assign_variable,Objective,VectorObjective
from tequila.hamiltonian import PauliString, QubitHamiltonian, paulis
from tequila.tools import list_assignment
from numpy import pi

from dataclasses import dataclass

# typing convenience shortcuts
UnionList = typing.Union[typing.Iterable[numbers.Integral], numbers.Integral]
UnionParam = typing.Union[Variable, FixedVariable]


class QGateImpl:
    """"
    BaseClass for internal gate representation
    All other gate classes should derive from here
    """

    @property
    def name(self):
        return self._name

    @property
    def target(self):
        return self._target

    @property
    def control(self):
        return self._control

    @property
    def qubits(self):
        # Set the active qubits
        if self.control:
            qubits = self.target + self.control
        else:
            qubits = self.target
        return sorted(tuple(set(qubits)))

    @property
    def max_qubit(self):
        return self.compute_max_qubit()

    def extract_variables(self):
        if self.is_parametrized() and hasattr(self.parameter, "extract_variables"):
            return self.parameter.extract_variables()
        else:
            return []

    def is_parametrized(self) -> bool:
        return hasattr(self, "parameter")

    def make_generator(self, include_controls=False):
        if self.generator and include_controls and self.is_controlled():
            return paulis.Qm(self.control) * self.generator

        return self.generator

    def __init__(self, name, target: UnionList, control: UnionList = None, generator: QubitHamiltonian = None):
        self._name = name
        self._target = tuple(list_assignment(target))
        self._control = tuple(list_assignment(control))
        self.finalize()
        self.generator = generator

    def copy(self):
        return copy.deepcopy(self)

    def dagger(self):
        """
        :return: return the hermitian conjugate of the gate.
        """
        result=copy.deepcopy(self)
        result.generator *= -1.0
        return result

    def is_controlled(self) -> bool:
        """
        :return: True if the gate is controlled
        """
        if len(self.control) == 0:
            return False
        else:
            return True

    def is_single_qubit_gate(self) -> bool:
        """
        Convenience and easier to interpret in code
        :return: True if the Gate only acts on one qubit (not controlled)
        """
        return (not self.control) and (len(self.target) == 1)

    def finalize(self):
        if not self.target:
            raise Exception('Received no targets upon initialization')
        if self.is_controlled():
            for c in self.target:
                if c in self.control:
                    raise Exception("control and target are the same qubit: " + self.__str__())
        if hasattr(self,"generator") and self.generator:
            if set(list(self.generator.qubits)) != set(list(self.target)):
                raise Exception("qubits of generator and targets don't agree -- mapping error?\n gate = {}".format(self.__str__()))
        if hasattr(self, "generators"):
            genq = []
            for generator in self.generators:
                genq += generator.qubits
            if set(list(genq)) != set(list(self.target)):
                raise Exception("qubits of generator and targets don't agree -- mapping error?\n gate = {}".format(self.__str__()))


    def __str__(self):
        result = str(self.name) + "(target=" + str(self.target)
        if not self.is_single_qubit_gate():
            result += ", control=" + str(self.control)
        result += ")"
        return result

    def __repr__(self):
        """
        Todo: Add Nice stringification
        """
        return self.__str__()

    def compute_max_qubit(self):
        """
        :return: highest qubit index used by this gate
        """
        if self.control is None:
            return max(self.target)
        else:
            return max(self.target + self.control)

    def __eq__(self, other):
        if self.name != other.name:
            return False
        if self.target != other.target:
            return False
        if self.control != other.control:
            return False
        return True

    def map_qubits(self, qubit_map: dict):
        mapped = copy.deepcopy(self)
        mapped._target = tuple([qubit_map[i] for i in self.target])
        if self.control is not None:
            mapped._control = tuple([qubit_map[i] for i in self.control])
        if hasattr(self, "generator") and self.generator:
            mapped.generator = self.generator.map_qubits(qubit_map=qubit_map)
        if hasattr(self, "generators"):
            mapped.generators = [i.map_qubits(qubit_map=qubit_map) for i in self.generators]
        mapped.finalize()
        if hasattr(self, "generator"):
            mapped.generator = self.generator.map_qubits(qubit_map=qubit_map)
        return mapped

class ParametrizedGateImpl(QGateImpl, ABC):
    '''
    the base class from which all parametrized gates inherit. User defined gates, when implemented, are liable to be members of this class directly.
    '''

    @property
    def parameter(self):
        return self._parameter

    @parameter.setter
    def parameter(self, other):
        self._parameter = assign_variable(variable=other)

    def __init__(self, name, parameter: UnionParam, target: UnionList, control: UnionList = None,
                generator: QubitHamiltonian = None):
        super().__init__(name=name, target=target, control=control, generator=generator)
        # failsafe
        if isinstance(parameter, VectorObjective):
            raise TequilaException('Received VectorObjective {} as parameter. This is forbidden.'.format(parameter))
        self._parameter = assign_variable(variable=parameter)

    def __str__(self):
        result = str(self.name) + "(target=" + str(self.target)
        if not self.is_single_qubit_gate():
            result += ", control=" + str(self.control)

        result += ", parameter=" + str(self._parameter)
        result += ")"
        return result

    def __eq__(self, other):
        if not isinstance(other, ParametrizedGateImpl):
            return False
        if not super().__eq__(other):
            return False
        if self._parameter != other._parameter:
            return False
        return True

    def dagger(self):
        result = copy.deepcopy(self)
        result._parameter = assign_variable(-self.parameter)
        return result

class DifferentiableGateImpl(ParametrizedGateImpl):

    @property
    def eigenvalues_magnitude(self):
        return self._eigenvalues_magnitude

    def __init__(self,eigenvalues_magnitude=None, assume_real=False, *args, **kwargs):
        self._eigenvalues_magnitude=eigenvalues_magnitude
        super().__init__(*args, **kwargs)
        self.assume_real=assume_real

    def shifted_gates(self, r=None):
        """
        Default shift rule, override this for special strategies
        Returns
        -------
            List of Tuples: [(weight, shifted_gate), (weight, shifted_gate)]
            The gradient compiler will assemble this the following way
            <H>_U with U = AU(a)B --> \sum weight <H>_V , with V=A shifted_gate

            shifted_gate can also be a whole circuit (either as QCircuit object or a list of gates)
        """
        if r is None:
            r = self.eigenvalues_magnitude

        s =  pi / (4 * r)
        shift_a = self.parameter + s
        shift_b = self.parameter - s
        right = copy.deepcopy(self)
        right.parameter = shift_a
        left = copy.deepcopy(self)
        left.parameter = shift_b

        if self.is_controlled():
            # following https://doi.org/10.1039/D0SC06627C
            p0 = paulis.Qp(self.control) # Qp = |0><0|
            right2 = GeneralizedRotationImpl(angle=s, generator=p0, eigenvalues_magnitude=r/2)  # controls are in p0
            left2 = GeneralizedRotationImpl(angle=-s, generator=p0, eigenvalues_magnitude=r/2)  # controls are in p0
            if not self.assume_real:
                # 4-point shift rule of arxiv:2104.05695 would saves gates here
                return [(r/2, [right, right2]), (-r/2, [left , left2]), (r/2, [right , left2]), (-r/2, [left , right2])]
            else:
                return [(r, [right, right2]), (-r, [left , left2])]
        else:
            return [ (r, right), (-r, left) ]

    def finalize(self):
        super().finalize()

class RotationGateImpl(DifferentiableGateImpl):
    axis_to_string = {0: "x", 1: "y", 2: "z"}
    string_to_axis = {"x": 0, "y": 1, "z": 2}

    @staticmethod
    def get_name(axis):
        axis = RotationGateImpl.assign_axis(axis)
        return "R" + RotationGateImpl.axis_to_string[axis]

    @property
    def axis(self):
        return self._axis

    @axis.setter
    def axis(self, value):
        self._axis = self.assign_axis(value)

    def __ipow__(self, power, modulo=None):
        self.parameter *= power
        return self

    def __pow__(self, power, modulo=None):
        result = copy.deepcopy(self)
        result.parameter *= power
        return result

    def __init__(self, axis, angle, target: list, control: list = None, assume_real=False):
        assert (angle is not None)
        super().__init__(eigenvalues_magnitude=0.5, assume_real=assume_real, name=self.get_name(axis=axis), parameter=angle, target=target, control=control)
        self._axis = self.assign_axis(axis)
        self.generator = self.assign_generator(self.axis, self.target)

    @staticmethod
    def assign_axis(axis):
        if axis in RotationGateImpl.string_to_axis:
            return RotationGateImpl.string_to_axis[axis]
        elif hasattr(axis, "lower") and axis.lower() in RotationGateImpl.string_to_axis:
            return RotationGateImpl.string_to_axis[axis.lower()]
        else:
            assert (axis in [0, 1, 2])
            return axis

    @staticmethod
    def assign_generator(axis, qubits):
        if axis == 0:
            return sum(paulis.X(q) for q in qubits)
        if axis == 1:
            return sum(paulis.Y(q) for q in qubits)

        return sum(paulis.Z(q) for q in qubits)


class PhaseGateImpl(DifferentiableGateImpl):

    def __init__(self, phase, target: list, control: list = None):
        assert (phase is not None)
        super().__init__(eigenvalues_magnitude=0.5, name='Phase', parameter=phase, target=target, control=control)
        self.generator = paulis.Z(target) - paulis.I(target)

    def __pow__(self, power, modulo=None):
        result = copy.deepcopy(self)
        result.parameter *= power
        return result

class PowerGateImpl(ParametrizedGateImpl):
    """
    Attributes
    ---------
    power
        numeric type (fixed exponent) or hashable type (parametrized exponent)
    parameter
        power multiplied by pi
        to be consitent with exp(-i a/2 G) representation [a: gate.parameter, G: gate.generator]
    """

    @property
    def power(self):
        return self.parameter/pi

    def __init__(self, name, generator: QubitHamiltonian,  target: list, power, control: list = None):
        if generator is None:
            assert name is not None and name.upper() in ["X", "Y", "Z"]
            generator = QubitHamiltonian.from_string("{}({})".format(name.upper(), target))
        if name is None:
            assert generator is not None
            name = str(generator)
        super().__init__(name=name, parameter=power * pi, target=target, control=control, generator=generator)


class GeneralizedRotationImpl(DifferentiableGateImpl):
    """
    A gate which behaves like a generalized rotation
     - its generator only has two distinguishable eigenvalues
     - it is then differentiable by the shift rule
     - shift needs to be given upon initialization (otherwise its default is 1/2)
     - the generator will not be verified to fullfill the properties
     Compiling will be done in analogy to a trotterized gate with steps=1 as default

    The gate will act in the same way as rotations and exppauli gates
    exp(-i angle/2 generator)
    """

    @staticmethod
    def extract_targets(generator):
        targets = []
        for ps in generator.paulistrings:
            targets += [k for k in ps.keys()]
        return tuple(set(targets))

    def __init__(self, angle, generator, control=None, eigenvalues_magnitude=0.5, steps=1, assume_real=False):
        super().__init__(eigenvalues_magnitude=eigenvalues_magnitude, assume_real=assume_real, name="GenRot", parameter=angle, target=self.extract_targets(generator), control=control)
        self.steps = steps
        self.generator = generator

class ExponentialPauliGateImpl(DifferentiableGateImpl):
    """
    Same convention as for rotation gates:
    Exp(-i angle/2 * paulistring)
    """

    def __init__(self, paulistring: PauliString, angle: float, control: typing.List[int] = None):
        super().__init__(eigenvalues_magnitude=0.5, name="Exp-Pauli", target=tuple(t for t in paulistring.keys()), control=control, parameter=angle)
        self.paulistring = paulistring
        self.generator = QubitHamiltonian.from_paulistrings(paulistring)
        self.finalize()

    def __str__(self):
        result = str(self.name) + "(target=" + str(self.target)
        if not self.is_single_qubit_gate():
            result += ", control=" + str(self.control)

        result += ", parameter=" + str(self.parameter)
        result += ", paulistring=" + str(self.paulistring)
        result += ")"
        return result

    def map_qubits(self, qubit_map: dict):
        mapped = super().map_qubits(qubit_map=qubit_map)
        mapped.paulistring = self.paulistring.map_qubits(qubit_map)
        return mapped


class TrotterizedGateImpl(ParametrizedGateImpl):

    def __init__(self, generator: QubitHamiltonian,
                 angle: typing.Union[numbers.Real, Variable],
                 steps: int = 1,
                 control: typing.Union[list, int] = None,
                 threshold: numbers.Real = 0.0,
                 randomize: bool = True, **kwargs):
        """
        :param generators: list of generators
        :param angles: coefficients for each generator
        :param steps: Trotter Steps
        :param control: control qubits
        :param threshold: neglect terms in the given Hamiltonians if their coefficients are below this threshold
        Note that for steps==1 as well as len(generators)==1 this has no effect
        :param randomize: randomize the trotter decomposition of the PauliStrings in the generator
        """

        assert angle is not None
        assert generator is not None

        super().__init__(name="Trotterized", target=self.extract_targets(generator), control=control, generator=generator, parameter=angle)
        self._parameter = angle
        self.steps = steps
        self.threshold = threshold
        self.randomize = randomize
        self.finalize()

    def __str__(self):
        result = str(self.name) + "(target=" + str(self.target)
        if not self.is_single_qubit_gate():
            result += ", control=" + str(self.control)

        result += ", angle=" + str(self.angle)
        result += ", generator=" + str(self.generator)
        result += ")"
        return result

    @staticmethod
    def extract_targets(generator):
        targets = []
        for ps in generator.paulistrings:
            targets += [k for k in ps.keys()]
        return tuple(set(targets))

