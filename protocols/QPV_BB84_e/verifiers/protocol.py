from netsquid.components.qprocessor import QuantumProcessor, PhysicalInstruction
from netsquid.nodes import Node, Network
from netsquid.components import instructions as instr
from netsquid.qubits.qformalism import QFormalism
from collections import defaultdict
from QPV_BB84_e.custom_models.network_components import QuantumConnection, ClassicalConnection, ConnectionDirection
from QPV_BB84_e.verifiers.alice_protocol import AliceProtocol
from QPV_BB84_e.verifiers.bob_protocol import BobProtocol
from QPV_BB84_e.custom_models.quantum_gates import PreparationGate
from QPV_BB84_e.custom_models.error_models import PhotonGeneratorErrorModel, BeamSplitterErrorModel

import netsquid as ns
import numpy as np

"""
QPV_BB84_e.py

Author: Julian Verweij
Institution: University of Amsterdam
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DESCRIPTION:
This file contains an implementation of the verifiers in the QPV_BB84_e protocol. Any players that want
to partake in the protocol can do so by connecting to the verifiers through the provided methods.
"""


# Quantum gate times in nanoseconds.
INIT_TIME = 0
GATE_TIME = 0
MEASURE_TIME = .02

# Channel connection speeds in km/s.
CCONN_SPEED = 3e5
QCONN_SPEED = 2e5


class Protocol():
    """This is a class representation of the QPV_BB84_e protocol. When the 'run' method is called, two verifiers
    Alice (at position P_A) and Bob (at position P_B) are simulated. They will verify for some position P_v,
    by communicating with the player(s). A player can connect to the verifiers using the provided methods.

    :param n: The number of rounds to run the protocol for.
    :type n: int
    :param m: The parameter in the protocol giving the amount of bases to encode in.
    :type m: int
    :param P_A: The position of Alice on the real number line.
    :type P_A: float
    :param P_B: The position of Bob on the real number line.
    :type P_B: float
    :param P_v: The verification position on the real number line.
    :type P_v: float
    """
    def __init__(self, n, m, P_A, P_B, P_v):
        # Work in the density matrix formalism to allow for error modelling.
        ns.set_qstate_formalism(QFormalism.DM)

        self.__setup_network(P_A, P_B)

        # Distances to verification position and connection speeds.
        network_details = {'P_Ad2P_v': P_v - P_A,
                           'P_Bd2P_v': P_B - P_v,
                           'cconn_speed': CCONN_SPEED,
                           'qconn_speed': QCONN_SPEED
                           }

        # The expected quantum time of Charlie, based on the quantum processing time of Alice.
        c_quantum_time = GATE_TIME + MEASURE_TIME + .001

        alice = self.__alice['node']
        alice.cdata['results'] = defaultdict(list)
        alice.cdata['ans_count'] = 0
        alice.cdata['n'] = n
        alice.cdata['m'] = m
        alice.cdata['network'] = network_details
        alice.cdata['c_quantum_time'] = c_quantum_time
        alice.cdata['qubit_prep_time'] = INIT_TIME

        bob = self.__bob['node']
        bob.cdata['results'] = defaultdict(list)
        bob.cdata['ans_count'] = 0
        bob.cdata['n'] = n
        bob.cdata['m'] = m
        bob.cdata['network'] = network_details
        bob.cdata['c_quantum_time'] = c_quantum_time
        bob.cdata['alice_qubit_prep_time'] = INIT_TIME

        self.__verification_position = P_v

    @property
    def alice_position(self):
        """Returns the position of Alice (P_A).

        :return: Alice's position, P_A.
        :rtype: float
        """
        return self.__alice['pos']

    @property
    def bob_position(self):
        """Returns the position of Bob (P_B).

        :return: Alice's position, P_B.
        :rtype: float
        """
        return self.__bob['pos']

    @property
    def verification_position(self):
        """Returns the verification position (P_v).

        :return: The verification position, P_v.
        :rtype: float
        """
        return self.__verification_position

    def __create_processor(self, fidelity_loss=.005, prob_absorption=.3):
        """A private method that returns a quantum processor with the given specifications. The processor
        supports qubit initialisation, the X gate, and the preparation gate used in the QPV_BB84_e protocol.

        :param fidelity_loss: The amount of loss in fidelity for qubit initialisation. Defaults to `.005`.
        :type fidelity_loss: optional, float
        :param prob_absorption: the probability of absorption of a photon when travelling through a beam splitter.
            Defaults to `.3`.
        :type prob_absorption: optional, float

        :returns: A quantum processor object with the given noise/loss characteristics.
        :rtype: :class:`netsquid.components.qprocessor.QuantumProcessor`
        """
        initialisation_error_model = PhotonGeneratorErrorModel(fidelity_loss)
        operation_error_model = BeamSplitterErrorModel(prob_absorption)

        instructions = [
            PhysicalInstruction(instr.INSTR_INIT, duration=INIT_TIME, quantum_noise_model=initialisation_error_model),
            PhysicalInstruction(instr.INSTR_X, duration=GATE_TIME, quantum_noise_model=operation_error_model),
            PhysicalInstruction(PreparationGate(), duration=GATE_TIME,
                                quantum_noise_model=operation_error_model)
        ]

        return QuantumProcessor('QProcessor', num_positions=1, phys_instructions=instructions)

    def __add_network_connection(self, node1, node2, conn, label, port_name_node1, port_name_node2):
        """A private method that connects two nodes in the network used by the verifiers,
        given a connection that must be used to connect the nodes.

        :param node1: A dictionary with the position of the node (index `pos`) and the :class:`netsquid.nodes.Node`
            object (index `node`) of node 1.
        :type node1: dict
        :param node2: A dictionary with the position of the node (index `pos`) and the :class:`netsquid.nodes.Node`
            object (index `node`) of node 2.
        :type node2: dict
        :param conn: A connection object to be used to connect the nodes.
        :type conn: :class:`netsquid.connections.Connection`
        :param label: A label used to identify the connection.
        :type label: str
        :param port_name_node1: A name used to identify the port of node 1.
        :type port_name_node1: str
        :param port_name_node2: A name used to identify the port of node 2.
        :type port_name_node2: str

        :return: An ordered tuple containing the connecting port names of the nodes.
        :rtype: (str, str)
        """
        if node1['node'].name not in self.__network.nodes:
            self.__network.add_node(node1['node'])

        if node2['node'].name not in self.__network.nodes:
            self.__network.add_node(node2['node'])

        return self.__network.add_connection(node1['node'], node2['node'], conn, label=label,
                                             port_name_node1=port_name_node1, port_name_node2=port_name_node2)

    def __setup_network(self, P_A, P_B):
        """A private method that sets up the network of the verifiers. The verifiers are connected using
        a classical connection, allowing them to communicate during the protocol.

        :param P_A: The position of Alice on the real number line.
        :type P_A: float
        :param P_B: The position of Alice on the real number line.
        :type P_B: float
        """
        self.__alice = {'node': Node('Alice', qmemory=self.__create_processor()), 'pos': P_A}
        self.__bob = {'node': Node('Bob'), 'pos': P_B}

        self.__network = Network('QPVBB84_network')

        # Classical connection from Alice to Bob and back to set up the state and basis to send to the prover.
        cconn = ClassicalConnection(length=P_B - P_A, direction=ConnectionDirection.BIDIRECTIONAL)
        self.__add_network_connection(self.__alice, self.__bob, cconn, 'Alice2Bob_classical', 'c_bob', 'c_alice')

    def __create_connection(self, conn_type, direction, length):
        """A private method that creates a classical or quantum connection of a certain length and direction.

        :param conn_type: A string indicating the connection type, `classical` or `quantum`.
        :type conn_type: str
        :param direction: The directionality of the channel.
        :type direction: :class:`QPV_BB84_e.custom_models.network_components.ConnectionDirection`
        :param length: The length of the channel in kilometres.
        :type length: float

        :return: A connection object with the given specifications.
        :rtype: :class:`QPV_BB84_e.custom_models.network_components.ClassicalConnection` or
            :class:`QPV_BB84_e.custom_models.network_components.QuantumConnection`
        """
        if conn_type == 'classical':
            return ClassicalConnection(length=length, direction=direction)
        if conn_type == 'quantum':
            return QuantumConnection(length=length, direction=direction)

    def connect_two_nodes(self, node1, node2, connection, label, port_name_node1, port_name_node2):
        """Connects two nodes in the network used by the verifiers,
        given a connection type that must be used to connect the nodes.

        :param node1: A dictionary with the position of the node (index `pos`) and the :class:`netsquid.nodes.Node`
            object (index `node`) of node 1.
        :type node1: dict
        :param node2: A dictionary with the position of the node (index `pos`) and the :class:`netsquid.nodes.Node`
            object (index `node`) of node 2.
        :type node2: dict
        :param connection: A tuple containing the connection type (`classical` or 'quantum') and the directionality.
        :type connection: (str, :class:`QPV_BB84_e.custom_models.network_components.ConnectionDirection`)
        :param label: A label used to identify the connection.
        :type label: str
        :param port_name_node1: A name used to identify the port of node 1.
        :type port_name_node1: str
        :param port_name_node2: A name used to identify the port of node 2.
        :type port_name_node2: str

        :return: An ordered tuple containing the connecting port names of the nodes.
        :rtype: (str, str)
        """
        length = np.abs(node1['pos'] - node2['pos'])
        conn = self.__create_connection(*connection, length)

        return self.__add_network_connection(node1, node2, conn, label, port_name_node1, port_name_node2)

    def connect_to_verifier(self, node, verifier_name, connection, label, port_name_node, port_name_verifier):
        """Connects a node to a verifier (Alice or Bob), given a connection specification.

        :param node: A dictionary with the position of the node (index `pos`) and the :class:`netsquid.nodes.Node`
            object (index `node`) of the node.
        :type node: dict
        :param verifier_name: A string indicating which verifier to connect to, `Alice` or `Bob`.
        :type verifier_name: str
        :param connection: A tuple containing the connection type (`classical` or `quantum`) and the directionality.
        :type conn: (str, :class:`QPV_BB84_e.custom_models.network_components.ConnectionDirection`)
        :param label: A label used to identify the connection.
        :type label: str
        :param port_name_node: A name used to identify the port of the node.
        :type port_name_node: str
        :param port_name_verifier: A name used to identify the port of the verifier.
        :type port_name_verifier: str

        :raises ValueError: When for a player location P_p, the condition P_A < P_p < P_B does not hold.

        :return: An ordered tuple containing the connecting port names of the nodes.
        :rtype: (str, str)

        """
        if self.alice_position >= node['pos'] or self.bob_position <= node['pos']:
            raise ValueError('The location of a player must be between P_A and P_B.')

        verifier = None
        length = None

        if verifier_name == 'Alice':
            verifier = self.__alice
            length = node['pos'] - self.alice_position
        elif verifier_name == 'Bob':
            verifier = self.__bob
            length = self.bob_position - node['pos']

        conn = self.__create_connection(*connection, length)

        return self.__add_network_connection(verifier, node, conn, label, port_name_node, port_name_verifier)

    def run(self):
        """Runs the QPV_BB84_e protocol.

        :return: The simulation statistics, the results of Alice, and the results of Bob.
        :rtype: list
        """
        protocol_alice = AliceProtocol(self.__alice['node'])
        protocol_bob = BobProtocol(self.__bob['node'])

        protocol_alice.start()
        protocol_bob.start()

        stats = ns.sim_run()

        return stats, self.__alice['node'].cdata['results'], self.__bob['node'].cdata['results']
