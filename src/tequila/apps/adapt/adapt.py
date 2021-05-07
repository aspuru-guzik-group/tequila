# Generalized Adaptive Solvers
# as described in Kottmann, Anand, Aspuru-Guzik: https://doi.org/10.1039/D0SC06627C

from tequila import QCircuit, QubitHamiltonian, gates, paulis, grad, simulate, TequilaWarning, TequilaException, minimize, ExpectationValue
import numpy
import dataclasses
import warnings
from itertools import combinations


@dataclasses.dataclass
class AdaptParameters:

    optimizer_args: dict = dataclasses.field(default_factory=lambda : {"method":"bfgs"})
    compile_args: dict = dataclasses.field(default_factory=lambda : {})
    maxiter:int = 100
    batch_size = 1
    energy_convergence: float = None
    gradient_convergence: float = 1.e-2
    max_gradient_convergence: float = 0.0
    degeneracy_threshold: float = 1.e-4

    def __str__(self):
        info = ""
        for k,v in self.__dict__.items():
            info += "{:30} : {}\n".format(k, v)
        return info

class AdaptPoolBase:
    """
    Standard class for operator pools in Adapt
    The pool is a list of generators (tequila QubitHamiltonians)
    """

    generators: list = None

    __n: int = 0 # for iterator, don't touch

    def __init__(self, generators, trotter_steps=1):
        self.generators = generators
        self.trotter_steps=1

    def make_unitary(self, k, label) -> QCircuit:
        return gates.Trotterized(generators=[self.generators[k]], angles=[(str(k), label)], steps=self.trotter_steps)

    def __iter__(self):
        self.__n = 0
        return self

    def __next__(self):
        if self.__n < len(self.generators):
            result = self.__n
            self.__n +=1
            return result
        else:
            raise StopIteration

    def __str__(self):
        return "{} with {} Generators".format(type(self).__name__, len(self.generators))

class ObjectiveFactoryBase:
    """
    Default class to create the objective in the Adapt solver
    This just creates the single ExpectationValue <H>_{Upre + U + Upost}
    and U will be the circuit that is adaptively constructed
    """

    Upre: QCircuit=QCircuit()
    Upost: QCircuit=QCircuit()
    H : QubitHamiltonian=None

    def __init__(self, H=None, Upre=None, Upost=None, *args, **kwargs):
        if H is None:
            raise TequilaException("No Hamiltonian was given to Adapt!")

        self.H = H
        if Upre is not None:
            self.Upre = Upre
        else:
            self.Upre = QCircuit()
        if Upost is not None:
            self.Upost = Upost
        else:
            self.Upost = QCircuit()

    def __call__(self, U, screening=False, *args, **kwargs):
        return ExpectationValue(H=self.H, U=self.Upre + U + self.Upost, *args, **kwargs)

    def grad_objective(self, *args, **kwargs):
        return self(*args, **kwargs)

    def __str__(self):
        return "{}".format(type(self).__name__)

class Adapt:

    operator_pool: AdaptPoolBase = None
    objective_factory = None
    parameters: AdaptParameters = AdaptParameters()


    def make_objective(self, U, variables=None, *args, **kwargs):
        return self.objective_factory(U=U, variables=variables, *args, **{**self.parameters.compile_args, **kwargs})

    def __init__(self, operator_pool, H=None,objective_factory=None, *args, **kwargs):
        self.operator_pool = operator_pool
        if objective_factory is None:
            self.objective_factory = ObjectiveFactoryBase(H, *args, **kwargs)
        else:
            self.objective_factory = objective_factory

        filtered = {k: v for k, v in kwargs.items() if k in self.parameters.__dict__}
        self.parameters = AdaptParameters(*args, **filtered)


    def __call__(self, static_variables = None, mp_pool=None, label=None, variables=None, *args, **kwargs):

        print("Starting Adaptive Solver")
        print(self)

        # count resources
        screening_cycles = 0
        objective_expval_evaluations = 0
        gradient_expval_evaluations = 0
        histories = []

        if static_variables is None:
            static_variables = {}

        if variables is None:
            variables = {**static_variables}
        else:
            variables = {**variables, **static_variables}

        U = QCircuit()

        initial_objective = self.make_objective(U, variables = variables)
        for k in initial_objective.extract_variables():
            if k not in variables:
                warnings.warn("variable {} of initial objective not given, setting to 0.0 and activate optimization".format(k), TequilaWarning)
                variables[k] = 0.0

        energy = simulate(initial_objective, variables=variables)
        for iter in range(self.parameters.maxiter):
            current_label = (iter,0)
            if label is not None:
                current_label = (iter, label)

            gradients = self.screen_gradients(U=U, variables=variables, mp_pool=mp_pool)

            grad_values = numpy.asarray(list(gradients.values()))
            max_grad = max(grad_values)
            grad_norm = numpy.linalg.norm(grad_values)

            if grad_norm < self.parameters.gradient_convergence:
                print("pool gradient norm is {:+2.8f}, convergence criterion met".format(grad_norm))
                break
            if numpy.abs(max_grad) < self.parameters.max_gradient_convergence:
                print("max pool gradient is {:+2.8f}, convergence criterion |max(grad)|<{} met".format(max_grad, self.parameters.max_gradient_convergence))
                break

            batch_size = self.parameters.batch_size

            # detect degeneracies
            degeneracies = [k for k in range(batch_size, len(grad_values))
                            if numpy.isclose(grad_values[batch_size-1],grad_values[k], rtol=self.parameters.degeneracy_threshold) ]

            if len(degeneracies) > 0:
                batch_size += len(degeneracies)
                print("detected degeneracies: increasing batch size temporarily from {} to {}".format(self.parameters.batch_size, batch_size))

            count = 0

            for k,v in gradients.items():
                Ux = self.operator_pool.make_unitary(k, label=current_label)
                U += Ux
                count += 1
                if count >= batch_size:
                    break

            variables = {**variables, **{k:0.0 for k in U.extract_variables() if k not in variables}}
            active_variables = [k for k in variables if k not in static_variables]

            objective = self.make_objective(U, variables=variables)
            result = minimize(objective=objective,
                                 variables=active_variables,
                                 initial_values=variables,
                                 **self.parameters.compile_args, **self.parameters.optimizer_args)

            diff = energy - result.energy
            energy = result.energy
            variables = result.variables

            print("-------------------------------------")
            print("Finished iteration {}".format(iter))
            print("current energy : {:+2.8f}".format(energy))
            print("difference     : {:+2.8f}".format(diff))
            print("grad_norm      : {:+2.8f}".format(grad_norm))
            print("max_grad       : {:+2.8f}".format(max_grad))
            print("circuit size   : {}".format(len(U.gates)))

            screening_cycles += 1
            mini_iter=len(result.history.extract_energies())
            gradient_expval = sum([v.count_expectationvalues() for k, v in grad(objective).items()])
            objective_expval_evaluations += mini_iter*objective.count_expectationvalues()
            gradient_expval_evaluations += mini_iter*gradient_expval
            histories.append(result.history)

            if self.parameters.energy_convergence is not None and numpy.abs(diff) < self.parameters.energy_convergence:
                print("energy difference is {:+2.8f}, convergence criterion met".format(diff))
                break

            if iter == self.parameters.maxiter - 1:
                print("reached maximum number of iterations")
                break

        @dataclasses.dataclass
        class AdaptReturn:
            U:QCircuit=None
            objective_factory:ObjectiveFactoryBase=None
            variables:dict=None
            energy: float = None
            histories: list = None
            screening_cycles: int = None
            objective_expval_evaluations: int =None
            gradient_expval_evaluations: int =None

        return AdaptReturn(U=U,
                           variables=variables,
                           objective_factory=self.objective_factory,
                           energy=energy,
                           histories=histories,
                           screening_cycles = screening_cycles,
                           objective_expval_evaluations=objective_expval_evaluations,
                           gradient_expval_evaluations=gradient_expval_evaluations)

    def screen_gradients(self, U, variables, mp_pool=None):

        args = []
        for k in self.operator_pool:
            arg = {}
            arg["k"] = k
            arg["variables"] = variables
            arg["U"] = U
            args.append(arg)

        if mp_pool is None:
            dEs = [self.do_screening(arg) for arg in args]
        else:
            print("screen with {} workers".format(mp_pool._processes))
            dEs = mp_pool.map(self.do_screening, args)
        dEs = dict(sorted(dEs, reverse=True, key=lambda x: numpy.fabs(x[1])))
        return dEs

    def do_screening(self, arg):
        Ux = self.operator_pool.make_unitary(k=arg["k"], label="tmp")
        Utmp = arg["U"] + Ux
        variables = {**arg["variables"]}
        objective = self.make_objective(Utmp, screening=True, variables=variables)


        dEs = []
        for k in Ux.extract_variables():
            variables[k] = 0.0
            dEs.append(grad(objective, k))

        gradients=[numpy.abs(simulate(objective=dE, variables=variables, **self.parameters.compile_args)) for dE in dEs]

        return arg["k"], sum(gradients)

    def __str__(self):
        result = str(self.parameters)
        result += str("{:30} : {}\n".format("operator pool: ", self.operator_pool))
        result += str("{:30} : {}\n".format("objective factory : ", self.objective_factory))
        return result

class MolecularPool(AdaptPoolBase):

    def __init__(self, molecule, indices:str):
        """

        Parameters
        ----------
        molecule:
            a tequila molecule object
        indices
            a list of indices defining UCC operations
            indices refer to spin-orbitals
            e.g. indices = [[(0,2),(1,3)], [(0,2)], [(1,3)]]
            can be a string for predefined pools supported are UpCCD, UpCCSD, UpCCGD, and UpCCGSD
        """
        self.molecule = molecule

        if isinstance(indices, str):
            if not "CC" in indices.upper():
                raise TequilaException("Pool of type {} not yet supported.\nCreate your own by passing the initialized indices".format(indices))

            generalized = True if "G" in indices.upper() else False
            paired = True if "P" in indices.upper() else False
            singles = True if "S" in indices.upper() else False
            doubles = True if "D" in indices.upper() else False

            indices = []
            if doubles: indices += self.make_indices_doubles(generalized=generalized, paired=paired)
            if singles: indices += self.make_indices_singles(generalized=generalized)

        indices = [tuple(k) for k in indices]
        super().__init__(generators=indices)


    def make_indices_singles(self, generalized=False):
        indices = []
        for p in range(self.molecule.n_electrons//2):
            for q in range(self.molecule.n_electrons//2, self.molecule.n_orbitals):
                indices.append([(2*p, 2*q)])
                indices.append([(2*p+1, 2*q+1)])
        if not generalized:
            return indices

        for p in range(self.molecule.n_orbitals):
            for q in range(p+1, self.molecule.n_orbitals):
                if [(2*p, 2*q)] in indices:
                    continue
                indices.append([(2*p, 2*q)])
                indices.append([(2*p+1, 2*q+1)])
        return self.sort_and_filter_unique_indices(indices)

    def make_indices_doubles(self, generalized=False, paired=True):
        indices = []
        for p in range(self.molecule.n_electrons//2):
            for q in range(self.molecule.n_electrons//2, self.molecule.n_orbitals):
                indices.append([(2*p, 2*q),(2*p+1, 2*q+1)])

        if not generalized:
            return indices

        for p in range(self.molecule.n_orbitals):
            for q in range(p+1, self.molecule.n_orbitals):
                idx = [(2*p, 2*q),(2*p+1, 2*q+1)]
                if idx in indices:
                    continue
                indices.append(idx)

        if not paired:
            indices += self.make_indices_doubles_all(generalized=generalized)

        return self.sort_and_filter_unique_indices(indices)

    def make_indices_doubles_all(self, generalized=False):
        singles = self.make_indices_singles(generalized=generalized)
        unwrapped = [x[0] for x in singles]
        # now make all combinations of singles
        indices = [x for x in combinations(unwrapped, 2)]
        return self.sort_and_filter_unique_indices(indices)

    def sort_and_filter_unique_indices(self, indices):
        # sort as: [[(a,b),(c,d),(e,f)...],...]with a<c, a<b, c<d
        sorted_indices = []
        for idx in indices:
            idx = tuple([tuple(sorted(pair)) for pair in idx]) # sort internal pairs (a<b, c<d, etc)
            # avoid having orbitals show up multiple times in excitatin strings
            idx = tuple([pair for pair in idx if sum([1 for pair2 in idx if pair[0] in pair2 or pair[1] in pair2 ])==1 ])
            if len(idx) == 0:
                continue
            idx = tuple(list(set(idx))) # avoid repetitions (like ((0,2),(0,2)))
            idx = tuple(sorted(idx, key=lambda x:x[0])) # sort pairs by first entry (a<c)
            sorted_indices.append(idx)
        return list(set(sorted_indices))



    def make_unitary(self, k, label):
        return self.molecule.make_excitation_gate(indices=self.generators[k], angle=(self.generators[k], label), assume_real=True)

class PseudoSingletMolecularPool(MolecularPool):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        indices = []
        for idx in self.generators:
            if len(idx) == 1:
                combined = ( ((idx[0][0]//2*2, idx[0][1]//2*2)), ((idx[0][0]//2*2+1, idx[0][1]//2*2+1)) )
                if combined not in indices:
                    indices.append(combined)
            else:
                indices.append(tuple([idx]))

        self.generators = list(set(indices))

    def make_unitary(self, k, label):
        U = QCircuit()
        for idx in self.generators[k]:
            combined_variable = self.generators[k][0]
            U += self.molecule.make_excitation_gate(indices=idx, angle=(combined_variable,label))
        return U

class ObjectiveFactorySequentialExcitedState(ObjectiveFactoryBase):

    def __init__(self, H, circuits: list, factors: list, *args, **kwargs):
        self.circuits = circuits
        self.factors = factors
        super().__init__(H=H, *args, **kwargs)

    def __call__(self, U,  *args, **kwargs):
        circuit = self.Upre + U + self.Upost
        objective = ExpectationValue(H=self.H, U=circuit)
        Qp = paulis.Qp(U.qubits)
        # get all overlaps
        for i,Ux in enumerate(self.circuits):
            S2 = ExpectationValue(H=Qp, U=circuit+Ux.dagger())
            objective += numpy.abs(self.factors[i])*S2
        return objective
