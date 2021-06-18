from netsquid.protocols.nodeprotocols import NodeProtocol
from netsquid.components.qprogram import QuantumProgram
from netsquid.components import instructions as instr
from netsquid.qubits import qubitapi as qapi
from QPV_BB84_e.custom_models.quantum_gates import PreparationGate

import netsquid as ns

"""
charlie_protocol.py

Author: Julian Verweij
Institution: University of Amsterdam
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DESCRIPTION:
This file contains the implementation of the protocol that Charlie, the honest prover, runs in the
QPV_BB84_e protocol.
"""


class MeasureProgram(QuantumProgram):
    """This is a class representation of the quantum program in which Charlie measures the qubit
    received from Alice in the basis provideded by Bob.
    """
    def program(self, m, m_0, m_1, physical):
        """Runs the quantum program on the qubit in register 0 in the quantum memory. Theta and phi are
        computed from m_0 and m_1, and then the corresponding (inverse) quantum gate is applied on the qubit
        before measurement in the standard basis. The measurement outcome label is 'c_i'.

        :param m: The m parameter in the matrix.
        :type m: int
        :param m_0: The ordered tuple received from Alice.
        :type m_0: (int, int)
        :param m_1: The message received from Bob.
        :type m_1: int
        :param physical: Whether to run the program on physical hardware (with noise and loss) or not.
        :type physical: bool
        """
        q, = self.get_qubit_indices(1)

        theta = (m_0[0] + m_1) % (2 * m + 1)
        phi = (m_0[1] + m_1) % (2 * m + 1)

        self.apply(PreparationGate(), q, theta=theta, phi=phi, m=m, inverse=True, physical=physical)

        self.apply(instr.INSTR_MEASURE, q, output_key='c_i', physical=physical)

        yield self.run()


class CharlieProtocol(NodeProtocol):
    """This is a class representation of the protocol that Charlie, the honest prover, runs during
    the protocol. The basic idea is that Charlie receives a qubit and classical information from Alice,
    and classical information from Bob. With the classical messages the basis can be determined, and
    the qubit can be measured. Then, the results are returned to Alice and Bob.
    """
    def setup_ports(self):
        """Set up the ports used to communicate classically and quantumly with the other parties.
        """
        self.q_port_alice = self.node.ports['q_alice']
        self.c_port_alice = self.node.ports['c_alice']
        self.c_port_bob = self.node.ports['c_bob']

    def process_m_0(self):
        """Extract the message m_0 from the port with Alice.
        """
        msg = self.c_port_alice.rx_input().items
        self.m_0 = msg[0][1]

    def process_m_1(self):
        """Extract the message m_1 from the port with Bob.
        """
        msg = self.c_port_bob.rx_input().items
        self.m_1 = msg[0][1]

    def measure_qubit(self):
        """Start the quantum measurement program with the values received from Alice
        and Bob.
        """
        self.measure_program = MeasureProgram()
        self.node.qmemory.execute_program(self.measure_program, m=self.m, m_0=self.m_0,
                                          m_1=self.m_1, physical=True)

        self.m_0 = None
        self.m_1 = None

    def process_measurement(self):
        """Process the measurement outcome, and send the results to Alice and Bob. If the qubit has been
        lost in the measurement process, we tell Alice and Bob that the photon was lost.
        """
        c_i, = self.measure_program.output['c_i']

        self.node.cdata['c_i'].append(None)

        q = self.node.qmemory.pop(0)[0]
        qapi.discard(q)

        if c_i is not None:
            self.c_port_alice.tx_output(('MEASUREMENT', c_i))
            self.c_port_bob.tx_output(('MEASUREMENT', c_i))
        else:
            self.c_port_alice.tx_output(('NO_PHOTON', None))
            self.c_port_bob.tx_output(('NO_PHOTON', None))

    def run(self):
        """Continuously check for messages from Alice or Bob.
        """
        self.m = self.node.cdata['m']
        self.setup_ports()

        received_m_0 = False
        received_m_1 = False
        result = False

        self.m_0 = None
        self.m_1 = None

        while True:
            # Check if we received a classical or quantum message and from whom.
            expr = yield (self.await_program(self.node.qmemory) |
                          (self.await_port_input(self.c_port_alice) | self.await_port_input(self.c_port_bob)))

            if expr.first_term.first_term.value:
                result = True
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

            if self.m_0 is not None and self.m_1 is not None:
                # Wait for a picosecond before checking whether we received the qubit, as
                # we do not know the order of the messages.
                yield self.await_timer(end_time=ns.sim_time() + .001)

                if not self.node.qmemory.peek(0)[0]:
                    self.c_port_alice.tx_output(('NO_PHOTON', None))
                    self.c_port_bob.tx_output(('NO_PHOTON', None))

                    self.m_0 = None
                    self.m_1 = None
                else:
                    self.measure_qubit()

            if result:
                result = False

                self.process_measurement()
