"""Compatibility imports for the linked vault_state bodies."""

from vault_machine import adapters

globals().update(adapters(__name__))
