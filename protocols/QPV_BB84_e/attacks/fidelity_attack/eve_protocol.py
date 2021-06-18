from netsquid.protocols.nodeprotocols import NodeProtocol
from QPV_BB84_e.custom_models.quantum_gates import PreparationGate

import numpy as np

"""
eve_protocol.py

Author: Julian Verweij
Institution: University of Amsterdam
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DESCRIPTION:
This file contains the implementation of the protocol that Eve, one of the adversaries, runs in the
QPV_BB84_e protocol.
"""


class EveProtocol(NodeProtocol):
    """This is a class representation of the protocol that Eve, one of the two adversaries along with Dave,
    runs during the QPV_BB84_e protocol. The idea is that Eve receives m_1 from Bob and forwards it to Dave.
    When Eve receives Dave's message
    at a later point in time, she can decide what value to send to Alice and Bob based on the fidelity
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
        """Set up the ports used to communicate classically with the other parties.
        """
        self.c_port_bob = self.node.ports['c_bob']
        self.c_port_dave = self.node.ports['c_dave']

    def process_m_1(self):
        """Extract the message m_1 from the port with Bob.
        """
        msg = self.c_port_bob.rx_input().items
        self.m_1 = msg[0][1]

        self.c_port_dave.tx_output(('m_1', self.m_1))

    def process_message_dave(self):
        r"""Process the message we received from Dave. If Dave did not receive a photon, we forward that
        to Bob. Else, we decide what result to send to Bob by comparing
        the fidelity between :math:`B_A | x \rangle` and :math:`B_V | x \rangle`, where :math:`x` is
        the measurement outcome, :math:`B_A` is the adversaries' gate and :math:`B_V`
        is that of the verifiers. We take into account the loss in fibre that an honest prover would
        have.
        """
        msg = self.c_port_dave.rx_input().items
        measurement, self.theta, self.phi, m_0 = msg[0][1]

        outcome = None

        if measurement == 'NO_PHOTON':
            self.c_port_bob.tx_output(('NO_PHOTON', None))
        else:
            ket_x = np.array([1, 0]) if not outcome else np.array([0, 1])

            theta = (m_0[0] + self.m_1) % (2 * self.m + 1)
            phi = (m_0[1] + self.m_1) % (2 * self.m + 1)

            # The qubit using the gate we (Dave and Eve) used.
            adv_qubit = np.array(PreparationGate().create_operator(self.theta, self.phi, self.m).arr @ ket_x)

            # The qubit using the gate the verifiers used.
            ver_qubit = np.array(PreparationGate().create_operator(theta, phi, self.m).arr @ ket_x)

            f = self.fidelity(adv_qubit, ver_qubit)

            if f < (1 - self.node.cdata['l_fraction']) / 2:
                outcome = (measurement + 1) % 2
            elif f > (1 + self.node.cdata['l_fraction']) / 2:
                outcome = measurement

            if outcome is not None:
                self.c_port_bob.tx_output(('MEASUREMENT', outcome))
            else:
                self.c_port_bob.tx_output(('NO_PHOTON', None))

        self.node.cdata['e_i'].append(outcome)

    def run(self):
        """Continuously check for messages from Dave or Bob.
        """
        self.m = self.node.cdata['m']
        self.setup_ports()

        received_m_1 = False
        received_data = False
        self.m_0 = None
        self.m_1 = None

        while True:
            expr = yield self.await_port_input(self.c_port_bob) | self.await_port_input(self.c_port_dave)

            if expr.first_term.value:
                received_m_1 = True
            elif expr.second_term.value:
                received_data = True

            if received_m_1:
                received_m_1 = False

                self.process_m_1()

            if received_data and self.m_1 is not None:
                received_data = False

                self.process_message_dave()

                self.m_1 = None
