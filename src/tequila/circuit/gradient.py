from tequila.circuit.compiler import Compiler
from tequila.objective.objective import Objective, ExpectationValueImpl, Variable,\
    assign_variable, identity, VectorObjective
from tequila import TequilaException
from tequila.simulators.simulator_api import compile
import typing
import copy
from numpy import pi
from tequila.autograd_imports import jax, __AUTOGRAD__BACKEND__


def grad(objective: typing.Union[Objective,VectorObjective], variable: Variable = None, no_compile=False, *args, **kwargs):

    '''
    wrapper function for getting the gradients of Objectives,ExpectationValues, Unitaries (including single gates), and Transforms.
    :param obj (QCircuit,ParametrizedGateImpl,Objective,ExpectationValue,Transform,Variable): structure to be differentiated
    :param variables (list of Variable): parameter with respect to which obj should be differentiated.
        default None: total gradient.
    return: dictionary of Objectives, if called on gate, circuit, exp.value, or objective; if Variable or Transform, returns number.
    '''

    if variable is None:
        # None means that all components are created
        variables = objective.extract_variables()
        result = {}

        if len(variables) == 0:
            raise TequilaException("Error in gradient: Objective has no variables")

        for k in variables:
            assert (k is not None)
            result[k] = grad(objective, k, no_compile=no_compile)
        return result
    else:
        variable = assign_variable(variable)

    if variable not in objective.extract_variables():
        return 0.0

    if no_compile:
        compiled = objective
    else:
        compiler = Compiler(multitarget=True,
                            trotterized=True,
                            hadamard_power=True,
                            power=True,
                            controlled_phase=True,
                            controlled_rotation=True,
                            gradient_mode=True)

        compiled = compiler(objective, variables=[variable])

    if variable not in compiled.extract_variables():
        raise TequilaException("Error in taking gradient. Objective does not depend on variable {} ".format(variable))

    if isinstance(objective, ExpectationValueImpl):
        return __grad_expectationvalue(E=objective, variable=variable)
    elif objective.is_expectationvalue():
        return __grad_expectationvalue(E=compiled.args[-1], variable=variable)
    elif isinstance(compiled, Objective) or isinstance(compiled, VectorObjective):
        return __grad_objective(objective=compiled, variable=variable)
    else:
        raise TequilaException("Gradient not implemented for other types than ExpectationValue and Objective.")


def __grad_objective(objective: typing.Union[Objective, VectorObjective], variable: Variable):
    if isinstance(objective, VectorObjective):
        return __grad_vector_objective(objective, variable)
    else:
        args = objective.args
        transformation = objective.transformation
        dO = None

        processed_expectationvalues = {}
        for i, arg in enumerate(args):
            if __AUTOGRAD__BACKEND__ == "jax":
                df = jax.grad(transformation, argnums=i)
            elif __AUTOGRAD__BACKEND__ == "autograd":
                df = jax.grad(transformation, argnum=i)
            else:
                raise TequilaException("Can't differentiate without autograd or jax")

            # We can detect one simple case where the outer derivative is const=1
            if transformation is None or transformation == identity:
                outer = 1.0
            else:
                outer = Objective(args=args, transformation=df)

            if hasattr(arg, "U"):
                # save redundancies
                if arg in processed_expectationvalues:
                    inner = processed_expectationvalues[arg]
                else:
                    inner = __grad_inner(arg=arg, variable=variable)
                    processed_expectationvalues[arg] = inner
            else:
                # this means this inner derivative is purely variable dependent
                inner = __grad_inner(arg=arg, variable=variable)

            if inner == 0.0:
                # don't pile up zero expectationvalues
                continue

            if dO is None:
                dO = outer * inner
            else:
                dO = dO + outer * inner

        if dO is None:
            raise TequilaException("caught None in __grad_objective")
        return dO


def __grad_vector_objective(objective: typing.Union[Objective,VectorObjective], variable: Variable):
    argsets = objective.argsets
    transformations = objective._transformations
    outputs = []
    for pos in range(len(objective)):
        args = argsets[pos]
        transformation = transformations[pos]
        dO = None

        processed_expectationvalues = {}
        for i, arg in enumerate(args):
            if __AUTOGRAD__BACKEND__ == "jax":
                df = jax.grad(transformation, argnums=i)
            elif __AUTOGRAD__BACKEND__ == "autograd":
                df = jax.grad(transformation, argnum=i)
            else:
                raise TequilaException("Can't differentiate without autograd or jax")

            # We can detect one simple case where the outer derivative is const=1
            if transformation is None or transformation == identity:
                outer = 1.0
            else:
                outer = Objective(args=args, transformation=df)

            if hasattr(arg, "U"):
                # save redundancies
                if arg in processed_expectationvalues:
                    inner = processed_expectationvalues[arg]
                else:
                    inner = __grad_inner(arg=arg, variable=variable)
                    processed_expectationvalues[arg] = inner
            else:
                # this means this inner derivative is purely variable dependent
                inner = __grad_inner(arg=arg, variable=variable)

            if inner == 0.0:
                # don't pile up zero expectationvalues
                continue

            if dO is None:
                dO = outer * inner
            else:
                dO = dO + outer * inner

        if dO is None:
            dO = Objective()
        outputs.append(dO)
    if len(outputs) == 1:
        return outputs[0]
    return outputs


def __grad_inner(arg, variable):
    '''
    a modified loop over __grad_objective, which gets derivatives
     all the way down to variables, return 1 or 0 when a variable is (isnt) identical to var.
    :param arg: a transform or variable object, to be differentiated
    :param variable: the Variable with respect to which par should be differentiated.
    :ivar var: the string representation of variable
    '''

    assert (isinstance(variable, Variable))
    if isinstance(arg, Variable):
        if arg == variable:
            return 1.0
        else:
            return 0.0
    elif isinstance(arg, ExpectationValueImpl):
        return __grad_expectationvalue(arg, variable=variable)
    elif hasattr(arg, "abstract_expectationvalue"):
        E = arg.abstract_expectationvalue
        dE = __grad_expectationvalue(E, variable=variable)
        return compile(dE, **arg._input_args)
    else:
        return __grad_objective(objective=arg, variable=variable)


def __grad_expectationvalue(E: ExpectationValueImpl, variable: Variable):
    '''
    implements the analytic partial derivative of a unitary as it would appear in an expectation value. See the paper.
    :param unitary: the unitary whose gradient should be obtained
    :param variables (list, dict, str): the variables with respect to which differentiation should be performed.
    :return: vector (as dict) of dU/dpi as Objective (without hamiltonian)
    '''

    hamiltonian = E.H
    unitary = E.U
    if not (unitary.verify()):
        raise TequilaException("error in grad_expectationvalue unitary is {}".format(unitary))

    # fast return if possible
    if variable not in unitary.extract_variables():
        return 0.0

    param_gates = unitary._parameter_map[variable]

    dO = Objective()
    for idx_g in param_gates:
        idx, g = idx_g
        dOinc = __grad_shift_rule(unitary, g, idx, variable, hamiltonian)
        dO += dOinc

    assert dO is not None
    return dO


def __grad_shift_rule(unitary, g, i, variable, hamiltonian):
    '''
    function for getting the gradients of directly differentiable gates. Expects precompiled circuits.
    :param unitary: QCircuit: the QCircuit object containing the gate to be differentiated
    :param g: a parametrized: the gate being differentiated
    :param i: Int: the position in unitary at which g appears
    :param variable: Variable or String: the variable with respect to which gate g is being differentiated
    :param hamiltonian: the hamiltonian with respect to which unitary is to be measured, in the case that unitary
        is contained within an ExpectationValue
    :return: an Objective, whose calculation yields the gradient of g w.r.t variable
    '''


    # possibility for overwride in custom gate construction
    if hasattr(g, "shifted_gates"):
        inner_grad=__grad_inner(g.parameter, variable)
        shifted = g.shifted_gates()
        dOinc = Objective()
        for x in shifted:
            w,g = x
            Ux = unitary.replace_gates(positions=[i], circuits=[g])
            wx = w*inner_grad
            Ex = Objective.ExpectationValue(U=Ux, H=hamiltonian)
            dOinc += wx*Ex
        return dOinc
    else:
        raise TequilaException('No shift found for gate {}\nWas the compiler called?'.format(g))

