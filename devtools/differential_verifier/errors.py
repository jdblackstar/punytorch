"""Verifier-specific error categories.

These exceptions intentionally separate malformed scenarios and unsupported
features from numerical-domain skips and failures in the candidate runtime.
"""


class VerifierError(Exception):
    """Base class for expected verifier control flow."""


class InvalidScenario(VerifierError):
    """The serialized graph is malformed or shape-incompatible."""


class UnsupportedFeature(VerifierError):
    """The scenario requests an operation or dtype outside the supported slice."""


class NumericalDomainSkip(VerifierError):
    """The graph is valid, but its values are unsuitable for a meaningful check."""


class GeneratorRejection(VerifierError):
    """A proposed generated case was rejected before candidate execution."""
