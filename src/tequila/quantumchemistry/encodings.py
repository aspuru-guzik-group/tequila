"""
Collections of Fermion-to-Qubit encodings known to tequila
Most are Interfaces to OpenFermion
"""
from tequila.circuit.circuit import QCircuit
from tequila.circuit.gates import X
from tequila.hamiltonian.qubit_hamiltonian import QubitHamiltonian
import openfermion

def known_encodings():
    # convenience for testing and I/O
    encodings= {
        "JordanWigner":JordanWigner,
        "BravyiKitaev":BravyiKitaev,
        "BravyiKitaevFast": BravyiKitaevFast,
        "BravyiKitaevTree": BravyiKitaevTree,
        "TaperedBravyiKitaev": TaperedBravyKitaev
    }
    # aliases
    encodings = {**encodings,
                 "ReorderedJordanWigner": lambda **kwargs: JordanWigner(up_then_down=True, **kwargs),
                 "ReorderedBravyiKitaev": lambda **kwargs: BravyiKitaev(up_then_down=True, **kwargs),
                 "ReorderedBravyiKitaevTree": lambda **kwargs: BravyiKitaevTree(up_then_down=True, **kwargs),
                 }
    return {k.replace("_","").replace("-","").upper():v for k,v in encodings.items()}

class EncodingBase:

    @property
    def name(self):
        prefix=""
        if self.up_then_down:
            prefix="Reordered"
        if hasattr(self, "_name"):
            return prefix+self._name
        else:
            return prefix+type(self).__name__

    def __init__(self, n_electrons, n_orbitals, up_then_down=False, *args, **kwargs):
        self.n_electrons = n_electrons
        self.n_orbitals = n_orbitals
        self.up_then_down = up_then_down

    def __call__(self, fermion_operator:openfermion.FermionOperator, *args, **kwargs) -> QubitHamiltonian:
        """
        :param fermion_operator:
            an openfermion FermionOperator
        :return:
            The openfermion QubitOperator of this class ecoding
        """
        if self.up_then_down:
            op = openfermion.reorder(operator=fermion_operator, order_function=openfermion.up_then_down, num_modes=2*self.n_orbitals)
        else:
            op = fermion_operator

        fop = self.do_transform(fermion_operator=op, *args, **kwargs)
        fop.compress()
        return self.post_processing(QubitHamiltonian.from_openfermion(fop))

    def post_processing(self, op, *args, **kwargs):
        return op

    def up(self, i):
        if self.up_then_down:
            return i
        else:
            return 2*i

    def down(self, i):
        if self.up_then_down:
            return i+self.n_orbitals
        else:
            return 2*i+1

    def do_transform(self, fermion_operator:openfermion.FermionOperator, *args, **kwargs) -> openfermion.QubitOperator:
        raise Exception("{}::do_transform: called base class".format(type(self).__name__))

    def map_state(self, state:list, *args, **kwargs) -> list:
        """
        Expects a state in spin-orbital ordering
        Returns the corresponding qubit state in the class encoding
        :param state:
            basis-state as occupation number vector in spin orbitals
            sorted as: [0_up, 0_down, 1_up, 1_down, ... N_up, N_down]
            with N being the number of spatial orbitals
        :return:
            basis-state as qubit state in the corresponding mapping
        """
        """Does a really lazy workaround ... but it works
        :return: Hartree-Fock Reference as binary-number

        Parameters
        ----------
        reference_orbitals: list:
            give list of doubly occupied orbitals
            default is None which leads to automatic list of the
            first n_electron/2 orbitals

        Returns
        -------

        """
        # default is a lazy workaround, but it workds
        n_qubits = 2 * self.n_orbitals

        spin_orbitals = sorted([i for i,x in enumerate(state) if int(x)==1])

        string = "1.0 ["
        for i in spin_orbitals:
            string += str(i) + "^ "
        string += "]"

        fop = openfermion.FermionOperator(string, 1.0)
        op = self(fop)
        from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
        wfn = QubitWaveFunction.from_int(0, n_qubits=n_qubits)
        wfn = wfn.apply_qubitoperator(operator=op)
        assert (len(wfn.keys()) == 1)
        key = list(wfn.keys())[0].array
        return key

    def hcb_to_me(self, *args, **kwargs):
        return None

    def __str__(self):
        return type(self).__name__

class JordanWigner(EncodingBase):
    """
    OpenFermion::jordan_wigner
    """

    def do_transform(self, fermion_operator:openfermion.FermionOperator, *args, **kwargs) -> openfermion.QubitOperator:
        return openfermion.jordan_wigner(fermion_operator, *args, **kwargs)

    def map_state(self, state:list, *args, **kwargs):
        state = state + [0]*(self.n_orbitals-len(state))
        result = [0]*len(state)
        if self.up_then_down:
            return [state[2*i] for i in range(self.n_orbitals)] + [state[2*i+1] for i in range(self.n_orbitals)]
        else:
            return state

    def hcb_to_me(self, *args, **kwargs):
        U = QCircuit()
        for i in range(self.n_orbitals):
            U += X(target=self.down(i), control=self.up(i))
        return U

class BravyiKitaev(EncodingBase):
    """
    Uses OpenFermion::bravyi_kitaev
    """

    def do_transform(self, fermion_operator:openfermion.FermionOperator, *args, **kwargs) -> openfermion.QubitOperator:
        return openfermion.bravyi_kitaev(fermion_operator, n_qubits=self.n_orbitals*2)


class BravyiKitaevTree(EncodingBase):
    """
    Uses OpenFermion::bravyi_kitaev_tree
    """

    def do_transform(self, fermion_operator:openfermion.FermionOperator, *args, **kwargs) -> openfermion.QubitOperator:
        return openfermion.bravyi_kitaev_tree(fermion_operator, n_qubits=self.n_orbitals*2)

class BravyiKitaevFast(EncodingBase):
    """
    Uses OpenFermion::bravyi_kitaev_tree
    """

    def do_transform(self, fermion_operator:openfermion.FermionOperator, *args, **kwargs) -> openfermion.QubitOperator:
        n_qubits = openfermion.count_qubits(fermion_operator)
        if n_qubits != self.n_orbitals*2:
            raise Exception("BravyiKitaevFast transformation currently only possible for full Hamiltonians (no UCC generators).\nfermion_operator was {}".format(fermion_operator))
        op = openfermion.get_interaction_operator(fermion_operator)
        return openfermion.bravyi_kitaev_fast(op)

class TaperedBravyKitaev(EncodingBase):
    """
    Uses OpenFermion::symmetry_conserving_bravyi_kitaev (tapered bravyi_kitaev_tree arxiv:1701.07072)
    Reduces Hamiltonian by 2 qubits
    See OpenFermion Documentation for more
    Does not work for UCC generators yet
    """
    def __init__(self, n_electrons, n_orbitals, active_fermions=None, active_orbitals=None, *args, **kwargs):
        if active_fermions is None:
            self.active_fermions = n_electrons
        else:
            self.active_fermions = active_fermions

        if active_orbitals is None:
            self.active_orbitals = n_orbitals*2 # in openfermion those are spin-orbitals
        else:
            self.active_orbitals = active_orbitals

        if "up_then_down" in kwargs:
            raise Exception("Don't pass up_then_down argument to {}, it can't be changed".format(type(self).__name__))
        super().__init__(n_orbitals=n_orbitals, n_electrons=n_electrons, up_then_down=False, *args, **kwargs)

    def do_transform(self, fermion_operator:openfermion.FermionOperator, *args, **kwargs) -> openfermion.QubitOperator:
        if openfermion.count_qubits(fermion_operator) != self.n_orbitals*2:
            raise Exception("TaperedBravyiKitaev not ready for UCC generators yet")
        return openfermion.symmetry_conserving_bravyi_kitaev(fermion_operator, active_orbitals=self.active_orbitals, active_fermions=self.active_fermions)

    def map_state(self, state:list, *args, **kwargs):
        non_tapered_trafo = BravyiKitaevTree(up_then_down=True, n_electrons=self.n_electrons, n_orbitals=self.n_orbitals)
        key = non_tapered_trafo.map_state(state=state, *args, **kwargs)
        n_qubits = self.n_orbitals*2
        active_qubits = [i for i in range(n_qubits) if i not in [n_qubits - 1, n_qubits // 2 - 1]]
        key = [key[i] for i in active_qubits]
        return key


