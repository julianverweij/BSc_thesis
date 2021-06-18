from netsquid.protocols.nodeprotocols import NodeProtocol
from netsquid.components.qprogram import QuantumProgram
from netsquid.components import instructions as instr
from QPV_BB84_e.custom_models.quantum_gates import PreparationGate

import random
import math
import netsquid as ns
import numpy as np

"""
alice_protocol.py

Author: Julian Verweij
Institution: University of Amsterdam
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DESCRIPTION:
This file contains the implementation of the protocol that Alice, one of the verifiers, runs in the
QPV_BB84_e protocol.
"""


class InitStateProgram(QuantumProgram):
    """This is a class representation of the quantum program in which Alice encodes the bit she has
    chosen.
    """
    def program(self, b, m, theta, phi, physical):
        r"""Runs the quantum program on the qubit in register 0 in the quantum memory.
        First the qubit is initiated in the :math:`| 0 \rangle` state, and the :math:`X` gate is
        applied or not, based on the parameter b.
        For the given theta and phi, the corresponding quantum gate is then applied on the qubit.

        :param b: The bit to encode, or whether to apply the :math:`X` gate or not.
        :type b: bool
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
        self.apply(instr.INSTR_INIT, q, physical=physical)

        if b:
            self.apply(instr.INSTR_X, q, physical=physical)

        self.apply(PreparationGate(), q, theta=theta, phi=phi, m=m, physical=physical)

        yield self.run()


class AliceProtocol(NodeProtocol):
    """This is a class representation of the protocol that Alice, one of the verifiers along with Bob,
    runs during the QPV_BB84_e protocol.

    :raises Exception: When no player is connected to our port.
    """
    def setup_ports(self):
        """Set up the ports used to communicate classically and quantumly with the other parties.
        """
        self.c_port_bob = self.node.ports['c_bob']

        for port in self.node.ports:
            if port[0] == 'q' and port != 'q_bob':
                self.q_port_player = self.node.ports[port]
            elif port[0] == 'c' and port != 'c_bob':
                self.c_port_player = self.node.ports[port]

        if not self.c_port_player or not self.q_port_player:
            raise Exception('No player connected to port.')

    def setup_timing_vals(self, network):
        """Computes all the necessary timing details for the protocol, such as the difference in
        time for our quantum/classical message to arrive at P_v versus Bob's message.
        """
        self.classical_delta_time_P_v = network['P_Ad2P_v'] / network['cconn_speed'] * 1e9
        self.quantum_delta_time_P_v = network['P_Ad2P_v'] / network['qconn_speed'] * 1e9

        self.bob_classical_delta_time_P_v = network['P_Bd2P_v'] / network['cconn_speed'] * 1e9
        self.bob_quantum_delta_time_P_v = network['P_Bd2P_v'] / network['qconn_speed'] * 1e9

        self.delta_send_time = self.bob_classical_delta_time_P_v - self.quantum_delta_time_P_v
        self.delta_send_time_classical = self.bob_classical_delta_time_P_v - self.classical_delta_time_P_v

        self.qubit_prep_time = self.node.cdata['qubit_prep_time']

    def send_values_to_bob(self):
        """Send our choice of bit and r to Bob, so that he can check the prover's correctness and
        send m_1.
        """
        self.r = random.randint(0, 2 * self.m)
        self.c_port_bob.tx_output(('VALUES', [self.b, self.r]))

    def choose_basis_and_bit(self):
        """Choose a random basis to use and bit to encode."""
        self.b = random.getrandbits(1)
        self.theta = random.randint(0, self.m - 1)
        self.phi = random.randint(0, np.round(2 * self.m * np.sin(np.arccos(2 * (self.theta / self.m) - 1))))

    def process_result(self, msg, results):
        """Process the result received from the prover. We check whether it was received within the
        expected time and whether it was correct. If we did not send a photon, we disregard the message.
        """
        time_took = ns.sim_time() - self.t_sent
        time_expected = 2 * self.classical_delta_time_P_v + self.node.cdata['c_quantum_time']

        # Account for small numerical errors with isclose()
        results['t_i'].append(time_took <= time_expected or math.isclose(time_took, time_expected))

        results['m_0_i'].append(self.theta - self.r % self.m)

        if msg[0] == 'NO_PHOTON' and not self.not_sent:
            results['r_i'].append(msg[0])
        elif not self.not_sent:
            r = msg[1]

            results['r_i'].append(r == self.b)
            self.node.cdata['ans_count'] += 1

        self.not_sent = False

    def prepare_qubit(self):
        """Start the qubit preparation program with the chosen basis values.
        """
        qubit_init_program = InitStateProgram()
        self.node.qmemory.execute_program(qubit_init_program, b=self.b, m=self.m, theta=self.theta,
                                          phi=self.phi, physical=True)

    def run(self):
        """Continuously check for messages from the connected player or Bob, until n instances of
        :math:`c_1` and :math:`c_2` have been recorded.
        """
        results = self.node.cdata['results']

        self.m = self.node.cdata['m']
        self.setup_ports()
        self.setup_timing_vals(self.node.cdata['network'])

        received_bob_ready = False
        received_result = False
        qubit_ready = False
        self.not_sent = False

        # Ensure that the qubit is ready when we expect Bob's message.
        self.choose_basis_and_bit()

        if (self.classical_delta_time_P_v + self.bob_classical_delta_time_P_v
                + self.delta_send_time >= self.qubit_prep_time - .001):
            self.send_values_to_bob()

        yield self.await_timer(end_time=ns.sim_time() + max(self.classical_delta_time_P_v
                                                            + self.bob_classical_delta_time_P_v
                               + self.delta_send_time - self.qubit_prep_time - .001, 0))
        self.prepare_qubit()

        if (self.classical_delta_time_P_v + self.bob_classical_delta_time_P_v
                + self.delta_send_time < self.qubit_prep_time - .001):
            yield self.await_timer(end_time=ns.sim_time() + self.qubit_prep_time - self.classical_delta_time_P_v
                                   - self.bob_classical_delta_time_P_v + .001)

            self.send_values_to_bob()

        while self.node.cdata['ans_count'] < self.node.cdata['n']:
            expr = yield ((self.await_program(self.node.qmemory) | self.await_port_input(self.c_port_bob)) |
                          self.await_port_input(self.c_port_player))

            if expr.first_term.first_term.value:
                qubit_ready = True
            elif expr.first_term.second_term.value:
                received_bob_ready = True
            elif expr.second_term.value:
                received_result = True

            if qubit_ready and received_bob_ready:
                qubit_ready = False
                received_bob_ready = False

                # Only send the qubit if it has not been lost in manipulation.
                if (self.node.qmemory.peek(0)[0].qstate):
                    self.q_port_player.tx_output(self.node.qmemory.pop(positions=0))
                else:
                    results['r_i'].append('NOT_SENT')
                    self.not_sent = True

                if max(0, self.delta_send_time_classical) - self.delta_send_time > 0:
                    self.send_time = ns.sim_time() + max(0, self.delta_send_time_classical) - self.delta_send_time

                    yield self.await_timer(end_time=self.send_time)

                self.t_sent = ns.sim_time()

                self.c_port_player.tx_output(('m_0', ((self.theta - self.r) % (2 * self.m + 1),
                                                      (self.phi - self.r) % (2 * self.m + 1))))

            if received_result:
                received_result = False

                msg = self.c_port_player.rx_input().items[0]
                self.process_result(msg, results)

                if self.delta_send_time_classical > 0:
                    self.send_time = ns.sim_time() + self.delta_send_time_classical

                    yield self.await_timer(end_time=self.send_time)

                # Ensure that the qubit is ready when we expect Bob's message.
                self.choose_basis_and_bit()

                if (self.classical_delta_time_P_v + self.bob_classical_delta_time_P_v
                        + self.delta_send_time >= self.qubit_prep_time - .001):
                    self.send_values_to_bob()

                yield self.await_timer(end_time=ns.sim_time() + max(self.classical_delta_time_P_v
                                                                    + self.bob_classical_delta_time_P_v
                                       + self.delta_send_time - self.qubit_prep_time - .001, 0))
                self.prepare_qubit()

                if (self.classical_delta_time_P_v + self.bob_classical_delta_time_P_v
                        + self.delta_send_time < self.qubit_prep_time - .001):
                    yield self.await_timer(end_time=ns.sim_time() + self.qubit_prep_time - self.classical_delta_time_P_v
                                           - self.bob_classical_delta_time_P_v + .001)

                    self.send_values_to_bob()
