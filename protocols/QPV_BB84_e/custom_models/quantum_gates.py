from netsquid.components import IGate, IMeasure

import numpy as np
import netsquid as ns

"""
quantum_gates.py

Author: Julian Verweij
Institution: University of Amsterdam
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DESCRIPTION:
This file contains the quantum gates used in the QPV_BB84_e protocol. These are the preparation gate
that Alice uses to encode the qubit, as well as a measurement gate that can handle qubit loss.
"""


class PreparationGate(IGate):
    r"""This is a class representation of the preparation gate used in the QPV_BB84_e protocol by Alice.
    The matrix representation of the gate is equal to:

    .. math::
        U = \begin{bmatrix}
              \cos(\frac{\sigma}{2}) & e^{i \delta} \sin(\frac{\sigma}{2})\\
              -e^{-i \delta} \sin(\frac{\sigma}{2}) & \cos(\frac{\sigma}{2})
            \end{bmatrix}
        \text{where } \delta = \frac{\phi \pi}{m \sin(\sigma)} \text{ and } \sigma = \cos^{-1}(\frac{2 \theta}{m} - 1).
    """
    def __init__(self):
        super().__init__('PreparationGate')

    def execute(self, quantum_memory, positions, theta, phi, m, inverse=False):
        r"""Returns a :class:`netsquid.qubits.Operator` object that represents the gate for the given parameters.

        :param quantum_memory: The quantum memory to execute on.
        :type quantum_memory: :class:`netsquid.components.qmemory.QuantumMemory`
        :param positions: A list of positions in the quantum memory to operate on.
        :type positions: list
        :param theta: The :math:`\theta` parameter in the matrix.
        :type theta: float
        :param phi: The :math:`\phi` parameter in the matrix.
        :type phi: float
        :param m: The m parameter in the matrix.
        :type m: int
        :param inverse: Whether to return :math:`U` or :math:`U^\dagger`. Defaults to `False`.
        :type inverse: optional, bool

        :return: A `pydynaa` event.
        :rtype: :class:`pydynaa.core.Event`
        """
        op = self.create_operator(theta, phi, m, inverse)

        # Do nothing if the qubit has been lost.
        return quantum_memory.operate(op, positions) if quantum_memory.peek(0)[0].qstate else None

    def create_operator(self, theta, phi, m, inverse=False):
        r"""Returns an operator object that represents the gate for the given parameters.

        :param theta: The :math:`\theta` parameter in the matrix.
        :type theta: float
        :param phi: The :math:`\phi` parameter in the matrix.
        :type phi: float
        :param m: The m parameter in the matrix.
        :type m: float

        :return: The operator object for :math:`U` with the given parameters.
        :rtype: :class:`netsquid.qubits.Operator`
        """
        sigma = np.arccos(2 * (theta / m) - 1)
        delta = ((phi * np.pi) / (m * np.sin(sigma))) if sigma != 0 else 0

        cos = np.cos(sigma / 2)
        sin = np.sin(sigma / 2)

        op = ns.qubits.Operator('PreparationGate', np.nan_to_num(np.array([[cos, np.e**(1j * delta) * sin],
                                                                           [-np.e**(-1j * delta) * sin, cos]])))
        if inverse:
            op = op.inv

        return op


class MeasurementGate(IMeasure):
    """This is a class representation of a quantum measurement gate that measures in the computational
    basis. It can handle qubit loss.
    """
    def __init__(self):
        super().__init__('measurement_op')

    def execute(self, quantum_memory, positions):
        """Returns a :class:`netsquid.qubits.Operator` object that represents the gate for the given parameters.

        :param quantum_memory: The quantum memory to execute on.
        :type quantum_memory: :class:`netsquid.components.qmemory.QuantumMemory`
        :param positions: A list of positions in the quantum memory to operate on.
        :type positions: list

        :return: The measurement results if the qubit state was not `None`, else `[None]`.
        :rtype: list
        """
        # Do nothing if the qubit has been lost.
        return super().execute(quantum_memory, positions) if quantum_memory.peek(0)[0].qstate else [None]
