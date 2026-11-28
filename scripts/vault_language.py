"""Compatibility imports for the linked vault_language bodies."""

from vault_machine import adapters

globals().update(adapters(__name__))
