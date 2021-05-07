import numpy
from tequila.objective import format_variable_dictionary
from tequila.tools.qng import evaluate_qng
import sys
"""
Define Containers for SciPy usage
"""



class _EvalContainer:
    """
    Container Class to access scipy and keep the optimization history.
    This class is used by the SciPy optimizer and should not be used elsewhere.

    Attributes
    ---------
    objective:
        the objective to evaluate.
    param_keys:
        the dictionary mapping parameter keys to positions in a numpy array.
    samples:
        the number of samples to evaluate objective with.
    save_history:
        whether or not to save, in a history, information about each time __call__ occurs.
    print_level
        dictates the verbosity of printing during call.
    N:
        the length of param_keys.
    history:
        if save_history, a list of energies received from every __call__
    history_angles:
        if save_history, a list of angles sent to __call__.


    """

    def __init__(self, objective, param_keys, passive_angles=None, samples=None, save_history=True,
                 print_level: int = 3):
        self.objective = objective
        self.samples = samples
        self.param_keys = param_keys
        self.N = len(param_keys)
        self.save_history = save_history
        self.print_level = print_level
        self.passive_angles = passive_angles
        if save_history:
            self.history = []
            self.history_angles = []

    def __call__(self, p, *args, **kwargs):
        """
        call a wrapped objective.
        Parameters
        ----------
        p: numpy array:
            Parameters with which to call the objective.
        args
        kwargs

        Returns
        -------
        numpy.array:
            value of self.objective with p translated into variables, as a numpy array.
        """

        angles = dict((self.param_keys[i], p[i]) for i in range(self.N))
        if self.passive_angles is not None:
            angles = {**angles, **self.passive_angles}
        vars = format_variable_dictionary(angles)
        E = self.objective(variables=vars, samples=self.samples)
        if self.print_level > 2:
            print("E={:+2.8f}".format(E), " angles=", angles, " samples=", self.samples)
        elif self.print_level > 1:
            print("E={:+2.8f}".format(E))
        if self.save_history:
            self.history.append(E)
            self.history_angles.append(angles)
        sys.stdout.flush()
        return numpy.float64(E)  # jax types confuses optimizers


class _GradContainer(_EvalContainer):
    """
    Container Class to access scipy and keep the optimization history.
    This class is used by the SciPy optimizer and should not be used elsewhere.
    see _EvalContainer for details.

    """

    def __call__(self, p, *args, **kwargs):
        """
        call the wrapped qng.

        Parameters
        ----------
        p: numpy array:
            Parameters with which to call gradient
        args
        kwargs

        Returns
        -------
        numpy.array:
            value of self.objective with p translated into variables, as a numpy array.
        """
        dO = self.objective
        dE_vec = numpy.zeros(self.N)
        memory = dict()
        variables = dict((self.param_keys[i], p[i]) for i in range(len(self.param_keys)))
        if self.passive_angles is not None:
            variables = {**variables, **self.passive_angles}
        for i in range(self.N):
            dE_vec[i] = dO[self.param_keys[i]](variables=variables, samples=self.samples)
            memory[self.param_keys[i]] = dE_vec[i]

        self.history.append(memory)
        return numpy.asarray(dE_vec, dtype=numpy.float64)  # jax types confuse optimizers


class _QngContainer(_EvalContainer):
    """
    Container Class to access scipy and keep the optimization history.
    This class is used by the SciPy optimizer and should not be used elsewhere.
    see _EvalContainer for details.

    Attributes
    ----------
    combos:
        the qng dictionaries to get some objective's qng.

    Methods
    -------
    evaluate_qng:
        evaluate the qng.
    """


    def __init__(self, combos, param_keys, passive_angles=None, samples=None, save_history=True):

        super().__init__(objective=None, param_keys=param_keys, passive_angles=passive_angles,
                         samples=samples, save_history=save_history)

        self.combos = combos

    def evaluate_qng(self, variables):
        """
        just a wrapper over the evaluate_qng function. see that function in tools/qng.py for details.

        Parameters
        ----------
        variables: dict
            the variables to call with.

        Returns
        -------
        numpy.array
            the evaluated qng as a vector of floats.
        """
        return evaluate_qng(self.combos, variables)

    def __call__(self, p, *args, **kwargs):
        """
        return the wrapped qng of some objective, evaluated.
        Parameters
        ----------
        p: numpy array:
            Parameters with which to call the qng.
        args
        kwargs

        Returns
        -------
        numpy.array:
            value of the qng of some object with p translated into variables, as a numpy array.
        """
        memory = dict()
        variables = dict((self.param_keys[i], p[i]) for i in range(len(self.param_keys)))
        if self.passive_angles is not None:
            variables = {**variables, **self.passive_angles}
        out = self.evaluate_qng(variables=variables)
        for i in range(self.N):
            memory[self.param_keys[i]] = out[i]
        self.history.append(memory)
        return numpy.asarray(out, dtype=numpy.float64)


class _HessContainer(_EvalContainer):
    """
    Container Class to access scipy and keep the optimization history.
    This class is used by the SciPy optimizer and should not be used elsewhere.
    see _EvalContainer for details.

    """

    def __call__(self, p, *args, **kwargs):
        """
        call the wrapped Hessian.

        Parameters
        ----------
        p: numpy array:
            Parameters with which to call the hessian
        args
        kwargs

        Returns
        -------
        numpy.array:
            value of the hessian with p translated into variables, as a numpy array.
        """

        ddO = self.objective
        ddE_mat = numpy.zeros(shape=[self.N, self.N])
        memory = dict()
        variables = dict((self.param_keys[i], p[i]) for i in range(len(self.param_keys)))
        if self.passive_angles is not None:
            variables = {**variables, **self.passive_angles}
        for i in range(self.N):
            for j in range(i, self.N):
                key = (self.param_keys[i], self.param_keys[j])
                value = ddO[key](variables=variables, samples=self.samples)
                ddE_mat[i, j] = value
                ddE_mat[j, i] = value
                memory[key] = value
        self.history.append(memory)
        return numpy.asarray(ddE_mat, dtype=numpy.float64)  # jax types confuse optimizers
