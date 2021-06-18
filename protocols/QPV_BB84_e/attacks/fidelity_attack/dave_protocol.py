from netsquid.protocols.nodeprotocols import NodeProtocol
from netsquid.components.qprogram import QuantumProgram
from netsquid.components import instructions as instr
from netsquid.qubits import qubitapi as qapi
from QPV_BB84_e.custom_models.quantum_gates import PreparationGate

import numpy as np
import random

"""
dave_protocol.py

Author: Julian Verweij
Institution: University of Amsterdam
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DESCRIPTION:
This file contains the implementation of the protocol that Dave, one of the adversaries, runs in the
QPV_BB84_e protocol.
"""


class MeasureProgram(QuantumProgram):
    """This is a class representation of the quantum program in which Dave applies the inverse
    of the random basis he has chosen.
    """
    def program(self, m, theta, phi, physical):
        r"""Runs the quantum program on the qubit in register 0 in the quantum memory.
        For the given theta and phi, the corresponding (inverse) quantum gate is applied on the qubit
        before measurement in the standard basis. The measurement outcome label is 'd_i'.

        :param m: The m parameter in the matrix.
        :type m: int
        :param theta: The :math:`\theta` parameter in the matrix.
        :type theta: float
        :param phi: The :math:`\phi` parameter in the matrix.
        :type phi: float
        :param physical: Whether to run the program on physical hardware (with noise and loss) or not.
        :type physical: bool
        """
        q, = self.get_qubit_indices(1)

        self.apply(PreparationGate(), q, theta=theta, phi=phi, m=m, inverse=True, physical=physical)

        self.apply(instr.INSTR_MEASURE, q, output_key='d_i', physical=physical)

        yield self.run()


class DaveProtocol(NodeProtocol):
    """This is a class representation of the protocol that Dave, one of the two adversaries along with Eve,
    runs during the QPV_BB84_e protocol. The idea is that Dave receives m_0 and the qubit from Alice. Then,
    Dave chooses a random basis and measures the received qubit in said basis, and sends the results to Eve.
    When Dave receives m_1
    at a later point in time from Eve, he can decide what value to send to Alice and Bob based on the fidelity
    that can now be computed.
    """
    def fidelity(self, psi, psi_prime):
        """Returns the fidelity between two pure quantum states in the ket formalism.

        :param psi: The first quantum state.
        :type psi: :class:`numpy.ndarray`
        :param psi_prime: The second quantum state.
        :type psi: :class:`numpy.ndarray`

        :return: The fidelity between the two states.
        :rtype: float
        """
        return np.abs(np.vdot(psi, psi_prime))**2

    def setup_ports(self):
        """Set up the ports used to communicate classically and quantumly with the other parties.
        """
        self.q_port_alice = self.node.ports['q_alice']
        self.c_port_alice = self.node.ports['c_alice']
        self.c_port_eve = self.node.ports['c_eve']

    def process_m_0(self):
        """Extract the message m_0 from the port with Alice. Since quantum communication is inherently
        slower than classical communication in the model,
        the qubit must have been lost if we did not receive it yet at this point in time. We tell
        Eve this so she can send that to Bob.
        """
        msg = self.c_port_alice.rx_input().items
        self.m_0 = msg[0][1]

        if not self.node.qmemory.peek(0)[0]:
            self.c_port_eve.tx_output(('DATA', ('NO_PHOTON', None, None, None)))
            self.last_no_photon = True

            self.m_0 = None

    def process_m_1(self):
        """Extract the message m_1 from the port with Eve. If we did not receive a photon before,
        we disregard the message and tell Alice that we did not recieve a photon.
        """
        if not self.last_no_photon:
            msg = self.c_port_eve.rx_input().items
            self.m_1 = msg[0][1]
        else:
            msg = self.c_port_eve.rx_input().items
            self.c_port_alice.tx_output(('NO_PHOTON', None))
            self.last_no_photon = False

    def measure_qubit(self):
        r"""Start the quantum measurement program with a random choice of basis(:math:`\theta` and :math:`\phi`).
        """
        self.theta = random.randint(0, self.m - 1)
        self.phi = random.randint(0, np.round(2 * self.m * np.sin(np.arccos(2 * (self.theta / self.m) - 1))))

        self.measure_program = MeasureProgram()
        self.node.qmemory.execute_program(self.measure_program, m=self.m, theta=self.theta,
                                          phi=self.phi, physical=True)

    def process_measurement(self):
        """Process the measurement outcome, and send the results to Eve along with the basis
        choice and m_0.
        """
        d_i, = self.measure_program.output['d_i']
        self.d_i = d_i

        # Send the measurement result, theta, phi, and m_0 to Eve.
        self.c_port_eve.tx_output(('DATA', (self.d_i, self.theta, self.phi, self.stored_m_0)))

        q = self.node.qmemory.pop(0)[0]
        qapi.discard(q)

    def send_result(self):
        r"""Once we have received m_1 from Eve, we decide what result to send to Alice by comparing
        the fidelity between :math:`B_A | x \rangle` and :math:`B_V | x \rangle`, where :math:`x` is
        the measurement outcome, :math:`B_A` is the adversaries' gate and :math:`B_V`
        is that of the verifiers. We take into account the loss in fibre that an honest prover would
        have.
        """
        outcome = None

        ket_x = np.array([1, 0]) if not self.d_i else np.array([0, 1])

        theta = (self.stored_m_0[0] + self.m_1) % (2 * self.m + 1)
        phi = (self.stored_m_0[1] + self.m_1) % (2 * self.m + 1)

        # The qubit using the gate we (Dave and Eve) used.
        adv_qubit = np.array(PreparationGate().create_operator(self.theta, self.phi, self.m).arr @ ket_x)

        # The qubit using the gate the verifiers used.
        ver_qubit = np.array(PreparationGate().create_operator(theta, phi, self.m).arr @ ket_x)

        f = self.fidelity(adv_qubit, ver_qubit)

        if f < (1 - self.node.cdata['l_fraction']) / 2:
            outcome = (self.d_i + 1) % 2
        elif f > (1 + self.node.cdata['l_fraction']) / 2:
            outcome = self.d_i

        if outcome is not None:
            self.c_port_alice.tx_output(('MEASUREMENT', outcome))
        else:
            self.c_port_alice.tx_output(('NO_PHOTON', None))

        self.node.cdata['d_i'].append(outcome)

    def run(self):
        """Continuously check for messages from Alice or Eve.
        """
        self.m = self.node.cdata['m']
        self.setup_ports()

        received_qubit = False
        received_m_0 = False
        received_m_1 = False
        result = False
        self.last_no_photon = False
        self.m_0 = None
        self.stored_m_0 = None
        self.m_1 = None
        self.d_i = None

        while True:
            # Check if we received a classical or quantum message and from whom.
            expr = yield ((self.await_program(self.node.qmemory) | self.await_port_input(self.q_port_alice)) |
                          (self.await_port_input(self.c_port_alice) | self.await_port_input(self.c_port_eve)))

            if expr.first_term.first_term.value:
                result = True
            elif expr.first_term.second_term.value:
                received_qubit = True
            elif expr.second_term.first_term.value:
                received_m_0 = True
            elif expr.second_term.second_term.value:
                received_m_1 = True

            if received_m_0:
                received_m_0 = False

                self.process_m_0()

            if received_m_1:
                received_m_1 = False

                self.process_m_1()

            if received_qubit and self.m_0 is not None:
                received_qubit = False

                self.measure_qubit()

                self.stored_m_0 = self.m_0
                self.m_0 = None

            if result:
                result = False

                self.process_measurement()

            if self.m_1 is not None and self.d_i is not None:
                self.send_result()

                self.m_1 = None
                self.d_i = None
