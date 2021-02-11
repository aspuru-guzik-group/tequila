"""
Export QCircuits as qpic files
https://github.com/qpic/qpic/blob/master/doc/qpic_doc.pdf
"""

from tequila.circuit.compiler import Compiler
from tequila.circuit import gates
from tequila.circuit import QCircuit
from tequila.tools import number_to_string

import subprocess
from shutil import which, move
from os import remove

system_has_qpic = which("qpic") is not None
system_has_pdflatex = which("pdflatex") is not None


def assign_name(parameter):
    if isinstance(parameter, tuple):
        return "\\theta"
    if hasattr(parameter, "extract_variables"):
        return str(parameter.extract_variables()).lstrip('[').rstrip(']')
    return str(parameter)


def export_to_qpic(circuit: QCircuit, filename=None, always_use_generators=True, decompose_control_generators=True, group_together=True, qubit_names=None, *args, **kwargs) -> str:
    result = ""
    # define tequila blue color
    result = "COLOR tq 0.03137254901960784 0.1607843137254902 0.23921568627450981\n"
    if group_together is True:
        group_together = "TOUCH"
    # define wires
    names = dict()
    if qubit_names is None:
        qubit_names = circuit.qubits
    if isinstance(qubit_names, str):
        qubit_names = [qubit_names for i in range(len(circuit.qubits))]
    for i,q in enumerate(circuit.qubits):
        name = "a" + str(q)
        names[q] = name
        result += name + " W " + str(qubit_names[i]) + "\n"

    for g in circuit.gates:
        
        if always_use_generators and g.make_generator(include_controls=decompose_control_generators) is not None:
                if len(ps) == 0: continue
                for k,v in ps.items():
                    result += " a{qubit} P:fill=tq  \\textcolor{{white}}{{{op}}} ".format(qubit=k, op=v.upper())
                if g.is_controlled() and not decompose_control_generators:
                    for c in g.control:
                        result += names[c] + " "
                result += "\n"
            if hasattr(group_together, "upper"):
                for t in circuit.qubits:
                    result += "a{} ".format(t)
                result += "{}\n".format(group_together.upper())

        else:
            if g.name.upper() in ["Exp-Pauli".upper(), "GenRot".upper()]:
                # represent ExpPaulis as generators
                for ps in g.generator.paulistrings:
                    if len(ps) == 0: continue
                    for k,v in ps.items():
                        result += " a{qubit} P:fill=tq  \\textcolor{{white}}{{{op}}} ".format(qubit=k, op=v.upper())
                    if g.is_controlled():
                        for c in g.control:
                            result += names[c] + " "
                    result += "\n"
            else:
                for t in g.target:
                    result += names[t] + " "
                if hasattr(g, "angle"):
                    result += " G $R_{" + g.axis_to_string[g.axis] + "}(" + assign_name(g.parameter) + ")$ width=" + str(
                        25 + 5 * len(assign_name(g.parameter))) + " "
                elif hasattr(g, "parameter") and g.parameter is not None:
                    result += " G $" + g.name + "(" + assign_name(g.parameter) + ")$ width=" + str(
                        25 + 5 * len(assign_name(g.parameter))) + " "
                else:
                    result += g.name + " "

                if g.is_controlled():
                    for c in g.control:
                        result += names[c] + " "

        result += "\n"

    if filename is not None:
        filenamex = filename
        if not filenamex.endswith(".qpic"):
            filenamex = filename + ".qpic"
        with open(filenamex, "w") as file:
            file.write(result)

    return result

def export_to(circuit: QCircuit,
              filename: str,
              always_use_generators:bool=True,
              decompose_control_generators:bool=True,
              group_together:bool=True,
              qubit_names:list=None,*args, **kwargs):
    """
    Parameters
    ----------
    circuit:
        the tequila circuit to export
    filename:
        filename.filetype, e.g. my_circuit.pdf, my_circuit.png (everything that qpic supports)
    always_use_generators:
        represent all gates with their generators
    decompose_control_generators:
        Decompose the controls to generators. Effective only in combination with always_use_generators=True.
    group_together:
        Keep PauliStrings from the same generator together. Effective only in combination with always_use_generators=True.
        possible values: False, True, 'TOUCH' and 'BARRIER'. True is the same as TOUCH.
        BARRIER will create a visible barrier in qpic
    args
    kwargs

    Returns
    -------

    """
    if not system_has_qpic:
        raise Exception("You need qpic in order to export circuits to pictures ---\n pip install qpic")
    if "." not in filename:
        raise Exception("export_to: No filetype given {}, expected something like {}.pdf".format(filename,filename))
    
    filename_tmp = filename.split(".")
    ftype = filename_tmp[-1]
    fname = "".join(filename_tmp[:-1])
    if len(filename_tmp) ==1:
        ftype = ".pdf"
        fname = filename

    compiled = Compiler(trotterized=True)(circuit)

    export_to_qpic(circuit=compiled,
                   filename=fname,
                   always_use_generators=always_use_generators,
                   decompose_control_generators=decompose_control_generators,
                   group_together=group_together,
                   qubit_names=qubit_names, *args, **kwargs)
    if ftype != "qpic":
        subprocess.call(["qpic", "{}.qpic".format(fname), "-f", ftype])

