import math
from pathlib import Path
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qpy
from qiskit.circuit import Gate
from qiskit.qasm3 import dump as qasm3_dump, dumps as qasm3_dumps
from qiskit.quantum_info import Statevector


class RXZGate(Gate):
    """Custom 2-qubit RXZ interaction gate: exp(-i * theta * X (x) Z / 2)."""

    def __init__(self, theta: float):
        super().__init__("rxz", 2, [theta])

    def _define(self) -> None:
        defn = QuantumCircuit(2)
        defn.rzx(self.params[0], 1, 0)
        self._definition = defn

    def inverse(self, annotated: bool = False) -> "RXZGate":
        return RXZGate(-self.params[0])

    def power(self, exponent: float) -> "RXZGate":
        return RXZGate(exponent * self.params[0])

    def __array__(self, dtype=None, copy: Optional[bool] = None) -> np.ndarray:
        if copy is False:
            raise ValueError("Unable to avoid copy while creating array.")
        theta = float(self.params[0])
        cos = math.cos(0.5 * theta)
        isin = 1j * math.sin(0.5 * theta)
        return np.array(
            [
                [cos, -isin, 0, 0],
                [-isin, cos, 0, 0],
                [0, 0, cos, isin],
                [0, 0, isin, cos],
            ],
            dtype=dtype,
        )


def verify_basis_state(decimal_val: int = 19) -> bool:
    """Verifies Qiskit tensor ordering against manual Kronecker product calculation."""
    state_0 = [1, 0]
    state_1 = [0, 1]

    qc = QuantumCircuit(5)
    qc.x(0)
    qc.x(1)
    qc.x(4)
    qiskit_sv = Statevector(qc)

    individual_states = [state_1, state_1, state_0, state_0, state_1]
    manual_sv = [1]
    for qubit_state in individual_states:
        manual_sv = np.kron(qubit_state, manual_sv)

    return (
        bool(manual_sv[decimal_val] == 1)
        and bool(qiskit_sv[decimal_val] == 1)
    )


def build_advanced_circuit() -> QuantumCircuit:
    """Constructs the multi-qubit system with custom gates and dynamic control flow."""
    qr = QuantumRegister(12, name="q")
    cr = ClassicalRegister(2, name="c")
    qc = QuantumCircuit(qr, cr, name="UpgradedQuantumSystem")

    for idx in range(5):
        qc.h(qr[idx])
        qc.cx(qr[idx], qr[idx + 5])

    qc.cx(qr[1], qr[7])
    qc.x(qr[8])
    qc.cx(qr[1], qr[9])
    qc.x(qr[7])
    qc.cx(qr[1], qr[11])
    
    qc.append(RXZGate(math.pi / 4), [qr[3], qr[4]])

    qc.swap(qr[6], qr[11])
    qc.swap(qr[6], qr[9])
    qc.swap(qr[6], qr[10])
    qc.x(qr[6])

    qc.h(qr[0])
    qc.cx(qr[0], qr[1])
    qc.measure(qr[0], cr[0])

    qc.h(qr[0])
    qc.cx(qr[0], qr[1])
    qc.measure(qr[0], cr[1])

    with qc.if_test((cr[0], 0)) as else_:
        qc.x(qr[2])
    with else_:
        qc.h(qr[2])
        qc.z(qr[2])

    return qc


def export_qasm3(qc: QuantumCircuit, output_dir: Optional[str] = "output") -> str:
    """Exports the circuit to an OpenQASM 3 string and optionally saves it to disk."""
    qasm_str = qasm3_dumps(qc)

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        qasm_file = out_path / f"{qc.name}.qasm"
        with open(qasm_file, "w") as f:
            f.write(qasm_str)
        print(f"OpenQASM 3 file saved to: {qasm_file}")

    return qasm_str


def export_circuit(
    qc: QuantumCircuit, output_dir: str = "output"
) -> Tuple[Path, Path]:
    """Serializes the circuit to QPY format and saves a visual diagram PNG."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    qpy_file = out_path / f"{qc.name}.qpy"
    png_file = out_path / f"{qc.name}.png"

    with open(qpy_file, "wb") as f:
        qpy.dump(qc, f)

    fig = qc.draw("mpl")
    fig.savefig(png_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return qpy_file, png_file


if __name__ == "__main__":
    assert verify_basis_state(19), "Statevector verification failed."

    circuit = build_advanced_circuit()
    
    # Export artifacts
    qpy_path, img_path = export_circuit(circuit)
    qasm_string = export_qasm3(circuit)

    print("\n--- Raw OpenQASM 3 Output ---")
    print(qasm_string)
