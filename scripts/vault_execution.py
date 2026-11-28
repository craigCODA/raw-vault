"""Compatibility imports for the linked vault_execution bodies."""

from vault_machine import adapters
from vault_machine import primitive_namespace

COMPARISONS = primitive_namespace()["COMPARISONS"]
NUMBER_TYPES = primitive_namespace()["NUMBER_TYPES"]

globals().update(adapters(__name__))
