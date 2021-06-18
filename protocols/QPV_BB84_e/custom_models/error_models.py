from netsquid.components.models.qerrormodels import QuantumErrorModel
from netsquid.qubits import qubitapi

import numpy as np

"""
error_models.py

Author: Julian Verweij
Institution: University of Amsterdam
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DESCRIPTION:
This file contains the various error models used to realistically implement the QPV_BB84_e protocol.
"""


class PhotonGeneratorErrorModel(QuantumErrorModel):
    r"""This is a class representation of the photon generator loss model. In this model we use the
    theoretical quantum depolarisation channel to simulate reality.

    :param fidelity_loss: The loss in fidelity from a perfect :math:`|0 \rangle` when generating the photon.
    :type fidelity: float
    :param \*\*kwargs: Keyword arguments.
    :type \*\*kwargs: dict
    """
    def __init__(self, fidelity_loss, **kwargs):
        super().__init__(**kwargs)

        self.fidelity_loss = fidelity_loss

    def error_operation(self, qubits, delta_time=0, **kwargs):
        r"""Performs the error operation on the qiven qubits.

        :param qubits: A tuple of qubits.
        :type qubits: (:class:`netsquid.qubits.qubit.Qubit`, :class:`netsquid.qubits.qubit.Qubit`)
        :param delta_time: The amount of time the qubits have spent on the component in nanoseconds.
            Defaults to `0`.
        :type delta_time: optional, float
        :param \*\*kwargs: Keyword arguments.
        :type \*\*kwargs: dict
        """
        for qubit in qubits:
            rho = qubit.qstate.qrepr.dm
            d = np.shape(rho)[0]

            # Calculate xi as described in the thesis.
            xi = (1 - 2 * self.fidelity_loss) / (2 * np.trace(rho @ rho) - 1)

            # Check for the complete positivity condition.
            assert(xi >= -(1/(d**2 - 1)) and xi <= 1)

            new_rho = xi * rho + ((1 - xi) / d) * np.eye(d)
            qubitapi.assign_qstate([qubit], new_rho)


class OpticalFibreErrorModel(QuantumErrorModel):
    r"""This is a class representation of the optical fibre loss model. In this model we use the
    theoretical quantum depolarisation channel to simulate reality. We also take into account the
    fact that the qubit has already lost fidelity due to the photon generator being imperfect.

    :param fidelity_loss: The amount of fidelity loss from a density matrix :math:`\rho` entering the channel.
    :type fidelity: float
    :param length: The length of the fibre connection in kilometres.
    :type length: float
    :param \*\*kwargs: Keyword arguments.
    :type \*\*kwargs: dict
    """
    def __init__(self, fidelity_loss, length, **kwargs):
        super().__init__(**kwargs)

        self.length = length
        self.fidelity_loss, self.fidelity_loss_length = fidelity_loss

    def error_operation(self, qubits, delta_time=0, **kwargs):
        r"""Performs the error operation on the qiven qubits.

        :param qubits: A tuple of qubits.
        :type qubits: (:class:`netsquid.qubits.qubit.Qubit`, :class:`netsquid.qubits.qubit.Qubit`)
        :param delta_time: The amount of time the qubits have spent on the component in nanoseconds.
            Defaults to `0`.
        :type delta_time: optional, float
        :param \*\*kwargs: Keyword arguments.
        :type \*\*kwargs: dict
        """
        for qubit in qubits:
            # Do nothing if the qubit has been lost.
            if not qubit.qstate:
                return

            # Calculate the value for xi_prime as described in the thesis.
            rho_prime = qubit.qstate.qrepr.dm
            rho_prime_sq = rho_prime @ rho_prime
            dim = np.shape(rho_prime)[0]

            a = np.trace(rho_prime_sq - (rho_prime / 2))
            b = np.trace(rho_prime / 2)
            c = np.linalg.det(rho_prime_sq - (rho_prime / 2))
            d = np.linalg.det(rho_prime / 2)
            f = self.fidelity_loss

            def xi_prime(sign):
                return np.real(((sign * 2*np.sqrt(a**2*d + b**2*c + 2*b*c*f - 2*b*c - 4*c*d + c*f**2 - 2*c*f + c)
                               + a * (-b) - a * f + a) / (a**2 - 4 * c)))

            def fidelity(xi_prime):
                return np.real(np.trace(xi_prime * (rho_prime_sq - (rho_prime / 2)) + (rho_prime / 2)) +
                               2 * np.sqrt(xi_prime**2 * np.linalg.det((rho_prime_sq - (rho_prime / 2)))
                                           + np.linalg.det((rho_prime / 2))))

            # Check whether we need the negative or positive square root in the calculation.
            xi_prime = xi_prime(1) if np.isclose(fidelity(xi_prime(1)), 1 - f) else xi_prime(-1)

            # Check for the complete positivity condition.
            assert(xi_prime >= -(1/(dim**2 - 1)) and xi_prime <= 1)

            # Change the parameter for the loss model according to the length of the channel.
            xi_prime = xi_prime**(self.length / self.fidelity_loss_length)

            new_rho = xi_prime * rho_prime + ((1 - xi_prime) / dim) * np.eye(dim)

            qubitapi.assign_qstate([qubit], new_rho)


class QubitLossModel(QuantumErrorModel):
    """This is a class representation of a probabilistic qubit loss model.

    :param prob_loss: The probability that the qubits are lost.
    :type prob_loss: float
    """
    def __init__(self, prob_loss):
        self.prob_loss = prob_loss

    def error_operation(self, qubits, delta_time=0, **kwargs):
        r"""Performs the error operation on the qiven qubits.

        :param qubits: A tuple of qubits.
        :type qubits: (:class:`netsquid.qubits.qubit.Qubit`, :class:`netsquid.qubits.qubit.Qubit`)
        :param delta_time: The amount of time the qubits have spent on the component in nanoseconds.
            Defaults to `0`.
        :type delta_time: optional, float
        :param \*\*kwargs: Keyword arguments.
        :type \*\*kwargs: dict
        """
        for i, qubit in enumerate(qubits):
            if qubit:
                self.lose_qubit(qubits, i, self.prob_loss)


class BeamSplitterErrorModel(QubitLossModel):
    r"""This is a class representation of the beam splitter error model. In this model,
    we probabilistically lose a qubit due to beam splitter absorption.

    :param prob_absorption: The probability that a qubit is lost.
    :type prob_absorption: float
    :param \*\*kwargs: Keyword arguments.
    :type \*\*kwargs: dict
    """
    def __init__(self, prob_absorption, **kwargs):
        super().__init__(prob_absorption, **kwargs)


class PhotonDetectorErrorModel(QubitLossModel):
    r"""This is a class representation of the photon detector error model. In this model,
    we probabilistically lose a qubit due to photon detectors being imperfect.

    :param efficiency: The probability that a qubit is successfully detected.
    :type efficiency: float
    :param \*\*kwargs: Keyword arguments.
    :type \*\*kwargs: dict
    """
    def __init__(self, efficiency, **kwargs):
        super().__init__(1 - efficiency, **kwargs)
