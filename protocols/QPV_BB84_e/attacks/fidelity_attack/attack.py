from netsquid.components.qprocessor import QuantumProcessor, PhysicalInstruction
from netsquid.components import instructions as instr
from netsquid.nodes import Node
from QPV_BB84_e.attacks.fidelity_attack.dave_protocol import DaveProtocol
from QPV_BB84_e.attacks.fidelity_attack.eve_protocol import EveProtocol
from QPV_BB84_e.verifiers.protocol import Protocol
from QPV_BB84_e.custom_models.network_components import ConnectionDirection
from QPV_BB84_e.custom_models.quantum_gates import PreparationGate

import argparse
import netsquid as ns

"""
attack.py

Author: Julian Verweij
Institution: University of Amsterdam
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DESCRIPTION:
This file contains an implementation of the fidelity attack performed by adversaries Dave and Eve on the
QPV_BB84_e protocol. We refer to the thesis to see how the attack works.
"""


class Attack():
    """This is a class representation of an attack by Dave and Eve based on fidelity on the QPV_BB84_e protocol.
    Refer to the thesis to see how this attack works.
    When instantiating this class, a :class:`QPV_BB84_e.verifiers.protocol.Protocol` object
    is instantiated, representing the verifiers. Dave and Eve can then interact with Alice and Bob
    to participate in the protocol.

    :param n: The number of rounds to run the protocol for.
    :type n: int
    :param m: The parameter in the protocol giving the amount of bases to encode in.
    :type m: int
    :param P_A: The position of Alice on the real number line.
    :type P_A: float
    :param P_B: The position of Bob on the real number line.
    :type P_B: float
    :param P_D: The position of Dave on the real number line.
    :type P_D: float
    :param P_E: The position of Eve on the real number line.
    :type P_E: float
    :param P_v: The verification position on the real number line.
    :type P_v: float
    :param prob_absorption: The probability of absorption of a photon when travelling through a beam splitter
        for Charlie. Defaults to `.3`.
    :type prob_absorption: optional, float
    :param detector_efficiency: The detection efficiency of the photon detector for Charlie. Defaults to `.96`.
    :type detector_efficiency: optional, float
    """
    def __init__(self, n, m, P_A, P_B, P_D, P_E, P_v, charlie_prob_absorption=.3, charlie_detector_efficiency=.96):
        self.model = Protocol(n, m, P_A, P_B, P_v)
        self.setup(P_D, P_E)

        # Calculate l_fraction as described in the thesis.
        l_fraction = 1 - ((1 - charlie_prob_absorption) * charlie_detector_efficiency
                          * 10**((-(P_v - P_A) * .18) / 10))

        dave = self.dave['node']
        # The results that Dave measures.
        dave.cdata['d_i'] = []
        dave.cdata['n'] = n
        dave.cdata['m'] = m
        dave.cdata['l_fraction'] = l_fraction

        eve = self.eve['node']
        # The results that Eve measures.
        eve.cdata['e_i'] = []
        eve.cdata['n'] = n
        eve.cdata['m'] = m
        eve.cdata['l_fraction'] = l_fraction

    def create_processor(self):
        """Returns a quantum processor. No error models are used, as we do not assume limitations for the adversaries.
        The processor supports the preparation gate used in the QPV_BB84_e protocol and measurement
        in the standard basis.

        :returns: A quantum processor object.
        :rtype: :class:`netsquid.components.qprocessor.QuantumProcessor`
        """
        instructions = [
            PhysicalInstruction(PreparationGate(), duration=0),
            PhysicalInstruction(instr.INSTR_MEASURE, duration=0)
        ]

        return QuantumProcessor('QProcessor', num_positions=1, phys_instructions=instructions)

    def setup(self, P_D, P_E):
        """Sets up Dave and Eve's end of the simulation. The adversaries is initiated and connected to Alice and Bob
        with the connections required to participate in the QPV_BB84_e protocol. They are also connected themselves,
        allowing them to cooperate in the attack.

        :param P_D: The position of Dave on the real number line.
        :type P_D: float
        :param P_E: The position of Eve on the real number line.
        :type P_E: float
        """
        self.dave = {'node': Node('Dave', qmemory=self.create_processor()), 'pos': P_D}
        self.eve = {'node': Node('Eve'), 'pos': P_E}

        # Classical connection from Alice to Dave and back to send the basis to Dave and the result of
        # the measurement to Alice.
        self.model.connect_to_verifier(self.dave, 'Alice', ['classical', ConnectionDirection.BIDIRECTIONAL],
                                       'Dave2Alice_classical', 'c_dave', 'c_alice')

        # Classical connection from Bob to Eve and back to send r to Eve and the result of
        # the measurement to Bob.
        self.model.connect_to_verifier(self.eve, 'Bob', ['classical', ConnectionDirection.BIDIRECTIONAL],
                                       'Bob2Eve_classical', 'c_eve', 'c_bob')

        # Classical connection from Dave to Eve and back to send the basis and measurement information.
        self.model.connect_two_nodes(self.dave, self.eve, ['classical', ConnectionDirection.BIDIRECTIONAL],
                                     'Dave2Eve_classical', 'c_eve', 'c_dave')

        # Quantum connection from Alice to Dave to send the quantum state to be measured.
        _, port_d = self.model.connect_to_verifier(self.dave, 'Alice', ['quantum', ConnectionDirection.A2B],
                                                   'Alice2Dave_quantum', 'q_charlie', 'q_alice')

        self.dave['node'].ports[port_d].forward_input(self.dave['node'].qmemory.ports['qin0'])

    def run(self):
        """Runs the QPV_BB84_e protocol with Dave and Eve partaking as adversaries employing the fidelity attack.

        :return: The simulation statistics, the results of Alice, and the results of Bob.
        :rtype: list
        """
        ns.sim_reset()

        dave_protocol = DaveProtocol(self.dave['node'])
        eve_protocol = EveProtocol(self.eve['node'])

        dave_protocol.start()
        eve_protocol.start()

        return self.model.run()


def main():
    # For debugging, run the protocol with Dave and Eve.
    parser = argparse.ArgumentParser(description="""QPV_BB84 simulation using NetSquid.
                                     Alice and Bob are the verifiers, Dave and Eve are the adversaries.""")
    parser.add_argument('iterations', metavar='n', type=int, help='The number of iterations (qubits).')
    parser.add_argument('bases', metavar='m', type=int, help='The number of bases to choose from.')
    parser.add_argument('positions', metavar='player positions', type=float, nargs=2,
                        help='The positions of Alice (P_A) and Bob (P_B) on the real number line.')
    parser.add_argument('adversaries', metavar='adversary positions', type=float, nargs=2,
                        help="""The positions of adversaries Dave (P_D) and Eve (P_E) on the real number line.""")
    parser.add_argument('v_pos', metavar='verification position (P_v)', type=float,
                        help='The position to verify for.')

    args = parser.parse_args()

    if args.positions[0] >= args.positions[1]:
        parser.error('It is required that P_A < P_B.')

    if args.adversaries:
        if ((args.positions[0] >= args.adversaries[0] or args.adversaries[1] >= args.positions[1]) or
                (args.positions[0] >= args.v_pos or args.v_pos >= args.adversaries[1])):
            parser.error('It is required that P_A < P_D < P_v < P_E < P_B.')

    P_A, P_B = args.positions
    P_D, P_E = args.adversaries

    attack = Attack(args.iterations, args.bases, P_A, P_B, P_D, P_E, args.v_pos)

    stats, alice_data, bob_data = attack.run()

    print(stats)
    print('Alice\'s data (correct / on time per round):')
    print(alice_data['r_i'], alice_data['t_i'])
    print('Bob\'s data (correct / on time per round):')
    print(bob_data['r_i'], bob_data['t_i'])


if __name__ == '__main__':
    main()
