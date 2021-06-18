from netsquid.components.cchannel import ClassicalChannel
from netsquid.components.qchannel import QuantumChannel
from netsquid.nodes import Connection
from netsquid.components.models.delaymodels import FibreDelayModel
from netsquid.components.models.qerrormodels import FibreLossModel
from netsquid.nodes import DirectConnection
from enum import Enum
from QPV_BB84_e.custom_models.error_models import OpticalFibreErrorModel

"""
network_components.py

Author: Julian Verweij
Institution: University of Amsterdam
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DESCRIPTION:
This file contains the classes that describe the connections used in the protocol that connect the players
(nodes).
"""


class ConnectionDirection(Enum):
    """This is an `Enum` representation of the directionality of a connection. It allows unidirectionality
    (A to B or B to A), as well as bidirectionality.
    """
    A2B = 0
    B2A = 1
    BIDIRECTIONAL = 2


class ClassicalConnection(DirectConnection):
    """This is a class representation of a classical connection to be used between two nodes. The connection
    speed is the speed of light.

    :param length: The length of the channel in kilometres.
    :type length: float
    :param name: An identifier for the connection. Defaults to `ClassicalConnection`.
    :type name: optional, str
    :param direction: The directionality of the channel. Defaults to `bidirectional`.
    :type direction: optional, :class:`QPV_BB84_e.custom_models.network_components.ConnectionDirection`
    :param models: Models to be used in the connection, see the documentation for
        :class:`netsquid.components.cchannel.ClassicalChannel`. Defaults to `None`.
    :type models: optional, dict
    """
    def __init__(self, length, name='ClassicalConnection', direction=ConnectionDirection.BIDIRECTIONAL, models=None):
        super().__init__(name=name)

        # Have standard radio wave properties when no models are given. We use the FibreDelayModel class,
        # but change c to 3e5 to simulate the speed of radio waves (speed of light).
        if not models:
            models = {'delay_model': FibreDelayModel(c=3e5)}

        if direction != ConnectionDirection.B2A:
            self.add_subcomponent(ClassicalChannel('Channel_A2B', length=length, models=models),
                                  forward_input=[('A', 'send')],
                                  forward_output=[('B', 'recv')])

        if direction != ConnectionDirection.A2B:
            self.add_subcomponent(ClassicalChannel('Channel_B2A', length=length, models=models),
                                  forward_input=[('B', 'send')],
                                  forward_output=[('A', 'recv')])


class QuantumConnection(Connection):
    """This is a class representation of a quantum connection to be used between two nodes. The connection
    speed is the speed of light in optical fibre.

    :param length: The length of the channel in kilometres.
    :type length: float
    :param name: An identifier for the connection. Defaults to `QuantumConnection`.
    :type name: optional, str
    :param direction: The directionality of the channel. Defaults to `bidirectional`.
    :type direction: optional, :class:`QPV_BB84_e.custom_models.network_components.ConnectionDirection`
    :param models: Models to be used in the connection, see the documentation for
        :class:`netsquid.components.qchannel.QuantumChannel`. Defaults to `None`.
    :type models: optional, dict
    :param p_loss_init: The probability of qubit loss as it enters the channel due to fibre coupling efficiency.
        Defaults to `.2`.
    :type p_loss_init: optional, float
    :param p_loss_length: The attenuation of photons in fibre in decibel per kilometre. Defaults to `.18`.
    :type p_loss_length: optional, float
    :param fidelity_loss: A tuple containing the fidelity loss due to the environment for a certain length
        of fibre. Defaults to `(.047, 50)`.
    :type fidelity_loss: optional, (float, float)
    """
    def __init__(self, length, name='QuantumConnection', direction=ConnectionDirection.BIDIRECTIONAL,
                 models=None, p_loss_init=.2, p_loss_length=.18, fidelity_loss=(.047, 50)):
        super().__init__(name=name)

        # Have standard fibre properties when no models are given.
        if not models:
            models = {
                'delay_model': FibreDelayModel(c=2e5),
                'quantum_loss_model': FibreLossModel(p_loss_init=p_loss_init, p_loss_length=p_loss_length),
                'quantum_noise_model': OpticalFibreErrorModel(fidelity_loss=fidelity_loss, length=length)
            }

        if direction != ConnectionDirection.B2A:
            self.add_subcomponent(QuantumChannel('QChannel_A2B', length=length, models=models),
                                  forward_input=[('A', 'send')],
                                  forward_output=[('B', 'recv')])

        if direction != ConnectionDirection.A2B:
            self.add_subcomponent(QuantumChannel('QChannel_B2A', length=length, models=models),
                                  forward_input=[('B', 'send')],
                                  forward_output=[('A', 'recv')])
