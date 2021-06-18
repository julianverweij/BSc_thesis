from netsquid.components.qprocessor import QuantumProcessor, PhysicalInstruction
from netsquid.nodes import Node
from QPV_BB84_e.custom_models.network_components import ConnectionDirection
from QPV_BB84_e.honest_player.charlie_protocol import CharlieProtocol
from QPV_BB84_e.custom_models.quantum_gates import PreparationGate, MeasurementGate
from QPV_BB84_e.verifiers.protocol import Protocol
from QPV_BB84_e.custom_models.error_models import BeamSplitterErrorModel, PhotonDetectorErrorModel

import argparse
import netsquid as ns

"""
charlie.py

Author: Julian Verweij
Institution: University of Amsterdam
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DESCRIPTION:
This file contains an implementation of an honest player Charlie in the QPV_BB84_e protocol.
"""


class Charlie():
    """This is a class representation of the honest player Charlie in the QPV_BB84_e protocol.
    When instantiating this class, a :class:`QPV_BB84_e.verifiers.protocol.Protocol` object
    is instantiated, representing the verifiers. Charlie can then interact with Alice and Bob
    to participate in the protocol.

    :param n: The number of rounds to run the protocol for.
    :type n: int
    :param m: The parameter in the protocol giving the amount of bases to encode in.
    :type m: int
    :param P_A: The position of Alice on the real number line.
    :type P_A: float
    :param P_C: The position of Charlie on the real number line.
    :type P_C: float
    :param P_B: The position of Bob on the real number line.
    :type P_B: float
    :param P_v: The verification position on the real number line.
    :type P_v: float
    """
    def __init__(self, n, m, P_A, P_C, P_B, P_v):
        self.model = Protocol(n, m, P_A, P_B, P_v)
        self.setup(P_C)

        charlie = self.charlie['node']
        # The results that Charlie measures.
        charlie.cdata['c_i'] = []
        charlie.cdata['n'] = n
        charlie.cdata['m'] = m

    def create_processor(self, prob_absorption=.3, detector_efficiency=.96):
        """Returns a quantum processor with the given specifications. The processor
        supports the preparation gate used in the QPV_BB84_e protocol and loss-resistant measurement
        in the standard basis.

        :param prob_absorption: The probability of absorption of a photon when travelling through a beam splitter.
            Defaults to `.3`.
        :type prob_absorption: optional, float
        :param detector_efficiency: The detection efficiency of the photon detector. Defaults to `.96`.
        :type detector_efficiency: optional, float

        :returns: A quantum processor object with the given loss characteristics.
        :rtype: :class:`netsquid.components.qprocessor.QuantumProcessor`
        """
        operation_error_model = BeamSplitterErrorModel(prob_absorption=prob_absorption)
        measurement_error_model = PhotonDetectorErrorModel(efficiency=detector_efficiency)

        instructions = [
            PhysicalInstruction(PreparationGate(), duration=0, quantum_noise_model=operation_error_model),
            PhysicalInstruction(MeasurementGate(), duration=.02, quantum_noise_model=measurement_error_model,
                                apply_q_noise_after=False)
        ]

        return QuantumProcessor('QProcessor', num_positions=1, phys_instructions=instructions)

    def setup(self, P_C):
        """Sets up Charlie's end of the simulation. Charlie is initiated and connected to Alice and Bob
        with the connections required to participate in the QPV_BB84_e protocol.

        :param P_C: The position of Charlie on the real number line.
        :type P_C: float
        """
        self.charlie = {'node': Node('Charlie', qmemory=self.create_processor()), 'pos': P_C}

        # Classical connection from Alice to Charlie and back to send the basis to Charlie and the result of
        # the measurement to Alice.
        self.model.connect_to_verifier(self.charlie, 'Alice', ['classical', ConnectionDirection.BIDIRECTIONAL],
                                       'Charlie2Alice_classical', 'c_charlie', 'c_alice')

        # Classical connection from Bob to Charlie and back to send r to Charlie and the result of
        # the measurement to Bob.
        self.model.connect_to_verifier(self.charlie, 'Bob', ['classical', ConnectionDirection.BIDIRECTIONAL],
                                       'Bob2Charlie_classical', 'c_charlie', 'c_bob')

        # Quantum connection from Alice to Charlie to send the quantum state to be measured.
        _, port_c = self.model.connect_to_verifier(self.charlie, 'Alice', ['quantum', ConnectionDirection.A2B],
                                                   'Alice2Charlie_quantum', 'q_charlie', 'q_alice')

        self.charlie['node'].ports[port_c].forward_input(self.charlie['node'].qmemory.ports['qin0'])

    def run(self):
        """Runs the QPV_BB84_e protocol with Charlie partaking as an honest player.

        :return: The simulation statistics, the results of Alice, and the results of Bob.
        :rtype: list
        """
        ns.sim_reset()

        protocol = CharlieProtocol(self.charlie['node'])
        protocol.start()

        return self.model.run()


def main():
    # For debugging, run the protocol with Charlie.
    parser = argparse.ArgumentParser(description="""QPV_BB84 simulation using NetSquid.
                                     Alice and Bob are the verifiers, Charlie is the prover.""")
    parser.add_argument('iterations', metavar='n', type=int, help='The number of iterations (qubits).')
    parser.add_argument('bases', metavar='m', type=int, help='The number of bases to choose from.')
    parser.add_argument('positions', metavar='player positions', type=float, nargs=3,
                        help='The positions of Alice (P_A), Charlie (P_C), and Bob (P_B) on the real number line.')
    parser.add_argument('v_pos', metavar='verification position (P_V)', type=float,
                        help='The position to verify for. In order to succeed, P_C should be equal P_V.')

    args = parser.parse_args()

    if ((args.positions[0] >= args.positions[1] or args.positions[1] >= args.positions[2]) or
            (args.positions[0] >= args.v_pos or args.v_pos >= args.positions[2])):
        parser.error('It is required that P_A < P_C < P_B and P_A < P_V < P_B.')

    P_A, P_C, P_B = args.positions

    charlie = Charlie(args.iterations, args.bases, P_A, P_C, P_B, args.v_pos)

    stats, alice_data, bob_data = charlie.run()

    print(stats)
    print('Alice\'s data (correct / on time per round):')
    print(alice_data['r_i'], alice_data['t_i'])
    print('Bob\'s data (correct / on time per round):')
    print(bob_data['r_i'], bob_data['t_i'])


if __name__ == '__main__':
    main()
