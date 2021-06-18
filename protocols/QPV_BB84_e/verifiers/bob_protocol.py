from netsquid.protocols.nodeprotocols import NodeProtocol

import math
import netsquid as ns

"""
bob_protocol.py

Author: Julian Verweij
Institution: University of Amsterdam
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DESCRIPTION:
This file contains the implementation of the protocol that Bob, one of the verifiers, runs in the
QPV_BB84_e protocol.
"""


class BobProtocol(NodeProtocol):
    """This is a class representation of the protocol that Bob, one of the verifiers along with Alice,
    runs during the QPV_BB84_e protocol.
    """
    def send_ready_to_alice(self):
        """Let Alice know that we are ready to start a round of the protocol.
        """
        self.c_port_alice.tx_output(('READY', None))

    def setup_ports(self):
        """Set up the ports used to communicate classically with the other parties.

        :raises Exception: When no player is connected to our port.
        """
        self.c_port_alice = self.node.ports['c_alice']

        for port in self.node.ports:
            if port[0] == 'c' and port != 'c_alice':
                self.c_port_player = self.node.ports[port]

        if not self.c_port_player:
            raise Exception('No player connected to port.')

    def setup_timing_vals(self, network):
        """Computes all the necessary timing details for the protocol, such as the difference in
        time for our classical message to arrive at P_v versus Alice's messages.
        """
        self.classical_delta_time_P_v = network['P_Bd2P_v'] / network['cconn_speed'] * 1e9
        self.quantum_delta_time_P_v = network['P_Bd2P_v'] / network['qconn_speed'] * 1e9

        self.alice_classical_delta_time_P_v = network['P_Ad2P_v'] / network['cconn_speed'] * 1e9
        self.alice_quantum_delta_time_P_v = network['P_Ad2P_v'] / network['qconn_speed'] * 1e9

        self.delta_send_time = self.alice_quantum_delta_time_P_v - self.classical_delta_time_P_v
        self.alice_delta_send_time = self.classical_delta_time_P_v - self.alice_quantum_delta_time_P_v
        self.delta_send_time_classical = self.alice_classical_delta_time_P_v - self.classical_delta_time_P_v

        self.qubit_prep_time = self.node.cdata['alice_qubit_prep_time']

    def process_result(self, msg, results):
        """Process the result received from the prover. We check whether it was received within the
        expected time and whether it was correct.
        """
        time_took = ns.sim_time() - self.t_sent
        time_expected = 2 * self.classical_delta_time_P_v + self.node.cdata['c_quantum_time']

        # Account for small numerical errors with isclose()
        results['t_i'].append(time_took <= time_expected or math.isclose(time_took, time_expected))

        results['m_1_i'].append(self.r)

        if msg[0] == 'NO_PHOTON':
            results['r_i'].append(msg[0])
        else:
            r = msg[1]

            results['r_i'].append(r == self.b)
            self.node.cdata['ans_count'] += 1

    def run(self):
        """Continuously check for messages from the connected player or Bob, until n instances of
        :math:`c_1` and :math:`c_2` have been recorded.
        """
        results = self.node.cdata['results']

        self.m = self.node.cdata['m']
        self.setup_ports()
        self.setup_timing_vals(self.node.cdata['network'])

        received_values = False
        received_result = False

        # Ensure that Alice has time to prepare our qubit before she receives our message.
        yield self.await_timer(end_time=ns.sim_time() + max(0, self.qubit_prep_time - self.classical_delta_time_P_v
                               - self.alice_classical_delta_time_P_v + .001))

        self.send_ready_to_alice()

        while self.node.cdata['ans_count'] < self.node.cdata['n']:
            expr = yield self.await_port_input(self.c_port_alice) | self.await_port_input(self.c_port_player)

            if expr.first_term.value:
                received_values = True
            elif expr.second_term.value:
                received_result = True

            if received_values:
                received_values = False

                msg = self.c_port_alice.rx_input().items
                self.b, self.r = msg[0][1]

                if self.delta_send_time > 0:
                    self.send_time = ns.sim_time() + self.delta_send_time

                    yield self.await_timer(end_time=self.send_time)

                self.t_sent = ns.sim_time()

                self.c_port_player.tx_output(('m_1', self.r))

            if received_result:
                received_result = False

                msg = self.c_port_player.rx_input().items[0]
                self.process_result(msg, results)

                if self.delta_send_time_classical > 0:
                    self.send_time = ns.sim_time() + self.delta_send_time_classical

                    yield self.await_timer(end_time=self.send_time)

                # Ensure that Alice has time to prepare our qubit before she receives our message.
                yield self.await_timer(max(0, self.qubit_prep_time - self.classical_delta_time_P_v
                                       - self.alice_classical_delta_time_P_v + .001))

                self.send_ready_to_alice()
