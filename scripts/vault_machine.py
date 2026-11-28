"""Execute linked Vault bodies. The trusted runtime/ directory contains the program.

An invocation only supplies JSON arguments/parameters and an existing context.
Each operation is resolved, bound, invoked and completed by the dispatch graph.
Fixture roots without a runtime directory inherit the installed program.
"""

import builtins
import hashlib
import json
import operator
import re
import tracemalloc
from collections import defaultdict, deque
from contextvars import ContextVar
from copy import deepcopy
from datetime import date, timedelta
from fractions import Fraction
from functools import lru_cache
from itertools import product
from math import prod
from pathlib import Path
from time import perf_counter_ns
from types import SimpleNamespace

from arithmetic import compile_equation, compile_value
from reasoning_links import load_rules
from vault_graph_compiler import BodyCompiler, identifier
from vault_identity import asset_path
from vault_state_access import (
    ACTIVE_STATE, StateAccess, copy_mapping, mapping_call, pointer_component,
    read_item, replace_state, state_overlay, state_patch, write_item,
)


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ContextVar("vault_graph_execution", default=(None, None))
DISPATCH = "body/vault_machine/dispatch"
MODULES = ("vault_execution", "vault_language", "vault_machine", "vault_reasoning", "vault_state", "vault_world")
BUILTINS = """
    Fraction IndexError KeyError ValueError ZeroDivisionError all any bool dict
    enumerate float int isinstance iter len list map max min next range set sorted
    str sum tuple type zip
""".split()
# A change to any trusted executable dependency changes the identity even when
# the data graph is byte-for-byte identical.
HOST_FILES = ("vault_machine.py", "vault_graph_compiler.py", "vault_state_access.py", "arithmetic.py", "reasoning_links.py",
              "response_links.py", "vault_identity.py", "vocabulary_links.py")


def _host_surface(owner, names):
    return SimpleNamespace(**{name: getattr(owner, name) for name in names.split()})


def _mark(kind, address):
    callback, _ = ACTIVE.get()
    if callback is not None:
        callback({"kind": kind, "address": address})


def _record(kind, address, **fields):
    callback, _ = ACTIVE.get()
    if callback is not None:
        def encode(value):
            if isinstance(value, Fraction):
                return str(value)
            raise TypeError(f"Unexpected evidence type {type(value).__name__}")
        # Freeze the observed value now. Later instructions may mutate it.
        event = json.loads(json.dumps(fields, default=encode, ensure_ascii=False, allow_nan=False))
        callback({"kind": kind, "address": address, **event})


def record_read(location, path, value, scope):
    components = path.split(".")
    tracker = ACTIVE_STATE.get()
    origin = tracker.containers.get(id(scope)) if tracker is not None else None
    if origin is not None and components[0] not in origin[2]:
        parts = [*origin[1], *components]
    else:
        root = scope.get(components[0]) if isinstance(scope, dict) else None
        binding = tracker.containers.get(id(root)) if tracker is not None else None
        if binding is not None:
            parts = [*binding[1], *components[1:]]
        else:
            parts = components if components[0] == "state" else ["scope", *components]
    state = parts[0] == "state"
    if state and tracker is not None:
        tracker.remember(value, tuple(parts))
    address = "/".join(pointer_component(part) for part in parts)
    _record("state.read" if state else "scope.read", address,
            source=location, path=".".join(parts) if state else path, value=value)
    return value


def record_cursor(location, cursor, instruction):
    _record("link.select", "link/" + pointer_component(cursor),
            source=location, cursor=cursor, instruction=instruction)


def observe(name, value):
    _, context = ACTIVE.get()
    reader = getattr(context, "observe", None)
    return value if reader is None else reader(name, value)


def read_rules(context, relative):
    path = asset_path(context.root, relative)
    return load_rules(path, read_bytes=lambda file: context.read_asset(file.relative_to(context.root).as_posix()))


def _arithmetic_call(function, *args, **kwargs):
    _, context = ACTIVE.get()
    primitive = getattr(context, "primitive", None)
    if primitive is None:
        return function(*args, **kwargs)
    return primitive(function, *args, **kwargs)


def start_allocation_trace():
    owned = not tracemalloc.is_tracing()
    if owned:
        tracemalloc.start()
    return owned


def stop_allocation_trace(owned):
    if owned:
        tracemalloc.stop()


def primitive_namespace():
    comparisons = {name: getattr(operator, name) for name in ("eq", "ne", "lt", "le", "gt", "ge")}
    namespace = {name: getattr(builtins, name) for name in BUILTINS if name != "Fraction"}
    namespace.update({
        "__builtins__": {}, "_graph_mark": _mark, "_arithmetic_call": _arithmetic_call, "Fraction": Fraction,
        "COMPARISONS": comparisons, "COMMON_COMPARISONS": dict(zip(
            ("=", "!=", "<", "<=", ">", ">="), comparisons.values(), strict=True)),
        "NEGATION": "logic:not", "NUMBER_TYPES": frozenset((int, float, Fraction)),
        "json": _host_surface(json, "loads dumps"), "re": _host_surface(re, "I fullmatch findall"),
        "hashlib": _host_surface(hashlib, "sha256"),
        "tracemalloc": _host_surface(tracemalloc, "is_tracing start stop get_traced_memory"),
        "start_allocation_trace": start_allocation_trace, "stop_allocation_trace": stop_allocation_trace,
        "date": date, "timedelta": timedelta, "perf_counter_ns": perf_counter_ns, "deque": deque,
        "defaultdict": defaultdict, "deepcopy": deepcopy, "compile_value": compile_value,
        "compile_equation": compile_equation, "prod": prod, "product": product, "read_rules": read_rules,
        "record_read": record_read, "record_cursor": record_cursor,
        "_read_item": read_item, "_write_item": write_item, "_mapping_call": mapping_call,
        "_copy_mapping": copy_mapping, "state_overlay": state_overlay, "state_patch": state_patch,
        "replace_state": replace_state,
        "observe": observe,
    })
    return namespace


def _fingerprint(records):
    digest = hashlib.sha256()
    for name, payload in records:
        for field in (name.encode("utf-8"), payload):
            digest.update(len(field).to_bytes(8, "big"))
            digest.update(field)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def load_program(payloads):
    bodies, catalog = {}, {}
    for module, payload in payloads:
        graph = json.loads(payload)
        if graph.get("format") != "vault-body-graph/1" or graph.get("module") != module:
            raise ValueError(f"Invalid runtime declaration {module!r}")
        for name, body in graph["bodies"].items():
            address = f"body/{module}/{identifier(name)}"
            bodies[address] = body
        catalog[module] = {name: f"body/{module}/{name}" for name in graph["exports"]}
        if not all(address in bodies for address in catalog[module].values()):
            raise ValueError(f"Missing exported body in {module!r}")
    if DISPATCH not in bodies:
        raise ValueError("Missing linked dispatch routine")
    symbols = {address: "_body_" + str(index) for index, address in enumerate(bodies)}
    namespace = primitive_namespace()
    functions = {}
    exported = {address for module in catalog.values() for address in module.values()}

    def invoke_body(address, *frame):
        if address not in exported:
            raise ValueError(f"Dispatch did not select an exported body: {address!r}")
        return functions[address](*frame)

    namespace["invoke_body"] = invoke_body
    for address, body in bodies.items():
        function = BodyCompiler(address, body, symbols).compile(namespace)
        functions[address] = function
        namespace[symbols[address]] = function
    return bodies, catalog, functions


class Machine:
    def __init__(self, root=ROOT, trace=None):
        folder = Path(root).resolve() / "runtime"
        if not folder.exists():
            folder = ROOT / "runtime"
        self._payloads = tuple((module, (folder / (module + ".vault")).read_bytes()) for module in MODULES)
        self._bodies, self._catalog, self._functions = load_program(self._payloads)
        self.trace = trace

    def has(self, module, operator):
        return operator in self._catalog.get(module, {})

    def body_address(self, module, operator):
        if not self.has(module, operator):
            raise ValueError(f"Unknown graph operation {module}/{operator}")
        return self._catalog[module][operator]

    def node(self, address):
        parts = address.split("/")
        body = self._bodies["/".join(parts[:3])]
        if len(parts) == 4:
            return deepcopy(body["nodes"][parts[3]])
        if len(parts) == 3:
            return deepcopy(body)
        raise KeyError(address)

    def fingerprint(self):
        scripts = Path(__file__).resolve().parent
        # Normalize checkout line endings in both program and bootstrap.
        records = [(name, data.replace(b"\r\n", b"\n")) for name, data in self._payloads]
        records += [(name, (scripts / name).read_bytes().replace(b"\r\n", b"\n")) for name in HOST_FILES]
        return _fingerprint(records)

    def execute(self, module, operator, arguments, parameters, context):
        if not self.has(module, operator):
            raise ValueError(f"Unknown graph operation {module}/{operator}")
        return self._call(DISPATCH, (self._catalog, module, operator, arguments, parameters, context),
                          context, arguments)

    def call_helper(self, module, name, *args, context=None):
        """Execute a trusted integration's term binding or condition in this revision."""
        return self._call(f"body/{module}/{name}", args, context)

    def _call(self, address, args, context, arguments=None):
        callback = self.trace if self.trace is not None else getattr(context, "graph_event", None)
        token = ACTIVE.set((callback, context))
        state_token = ACTIVE_STATE.set(StateAccess(arguments, _record) if callback is not None else None)
        try:
            return self._functions[address](*args)
        finally:
            ACTIVE_STATE.reset(state_token)
            ACTIVE.reset(token)

    def helper(self, module, name):
        """Bind a historic public Python helper directly to its stored body."""
        return self._functions[f"body/{module}/{name}"]


@lru_cache(maxsize=1)
def bundled_machine():
    return Machine(ROOT)


def adapters(module):
    machine = bundled_machine()
    names = [address.split("/")[-1] for address in machine._bodies if address.startswith(f"body/{module}/")]
    aliases = {name: machine.helper(module, name) for name in names}
    operators = {name: aliases[name] for name in machine._catalog[module]}
    return aliases | {"OPERATORS": operators}
