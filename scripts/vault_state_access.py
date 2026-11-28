"""Track field provenance in a single graph call without wrapping user data.

The identity table holds only state containers the program actually visits.
Overlay keys belong to the supplying arguments until a graph patch commits
them as state. Copying a state dictionary preserves its field addresses.
"""

from contextlib import contextmanager
from contextvars import ContextVar


ACTIVE_STATE = ContextVar("vault_state_access", default=None)
TURN_ORIGINS = ContextVar("vault_turn_origins", default=None)


@contextmanager
def turn_state(state):
    """Keep field addresses across the turn's condition, binding, and operation calls."""
    origins = {id(state): (state, ("state",), frozenset())}
    save = state.get("save")
    if isinstance(save, (dict, list)):
        origins[id(save)] = (save, ("state", "save"), frozenset())
    token = TURN_ORIGINS.set(origins)
    try:
        yield
    finally:
        TURN_ORIGINS.reset(token)


def pointer_component(value):
    return str(value).replace("~", "~0").replace("/", "~1")


class StateAccess:
    def __init__(self, arguments, emit):
        self.emit = emit
        turn = TURN_ORIGINS.get()
        self.containers = turn if turn is not None else {}
        self.arguments = arguments
        state = arguments.get("state") if isinstance(arguments, dict) else None
        if id(state) not in self.containers:
            self.remember(state, ("state",))

    def remember(self, obj, path, masked=frozenset()):
        if isinstance(obj, (dict, list)):
            # Retaining the object for this invocation prevents id reuse.
            self.containers[id(obj)] = (obj, path, masked)

    def field(self, kind, location, owner, key, value, **extra):
        if owner is self.arguments and key == "state" and not isinstance(value, (dict, list)):
            self.emit(kind, "state", source=location, path="state", value=value, **extra)
            return
        origin = self.containers.get(id(owner))
        if origin is None or key in origin[2]:
            return
        path = (*origin[1], str(key))
        self.remember(value, path)
        self.emit(kind, "/".join(map(pointer_component, path)), source=location,
                  path=".".join(path), value=value, **extra)

    def copy(self, original, result, masked=frozenset()):
        origin = self.containers.get(id(original))
        if origin is not None:
            self.remember(result, origin[1], origin[2] | masked)


def read_item(location, owner, key):
    value = owner[key]
    tracker = ACTIVE_STATE.get()
    if tracker is not None:
        tracker.field("state.read", location, owner, key, value)
    return value


def write_item(location, value, owner, key):
    owner[key] = value
    tracker = ACTIVE_STATE.get()
    if tracker is not None:
        tracker.field("state.write", location, owner, key, value)


def mapping_call(location, method, *args, **kwargs):
    tracker = ACTIVE_STATE.get()
    owner = method.__self__
    if tracker is None or owner is not tracker.arguments and id(owner) not in tracker.containers:
        return method(*args, **kwargs)
    if method.__name__ == "setdefault":
        key = args[0]
        found = key in owner
        result = method(*args, **kwargs)
        tracker.field("state.read" if found else "state.write", location, owner, key, result, found=found)
        return result
    result = method(*args, **kwargs)
    if method.__name__ == "get":
        key = args[0]
        tracker.field("state.read", location, owner, key, result, found=key in owner)
    elif method.__name__ == "copy":
        tracker.copy(owner, result)
    elif method.__name__ == "update":
        # Graph update instructions receive JSON objects. Iterate the supplied
        # fields so a write of the current value still has evidence.
        keys = dict.fromkeys(args[0]) if args and isinstance(args[0], dict) else {}
        for key in keys | kwargs:
            tracker.field("state.write", location, owner, key, owner[key])
    return result


def copy_mapping(function, *args, **kwargs):
    result = function(*args, **kwargs)
    tracker = ACTIVE_STATE.get()
    if tracker is not None and args:
        tracker.copy(args[0], result)
    return result


def state_overlay(state, supplied):
    result = state | supplied
    tracker = ACTIVE_STATE.get()
    if tracker is not None:
        tracker.copy(state, result, frozenset(supplied))
    return result


def state_patch(location, state, writes, cursor=None):
    result = state | writes
    tracker = ACTIVE_STATE.get()
    if tracker is not None:
        tracker.remember(result, ("state",))
        details = {"cursor": cursor} if cursor is not None else {}
        for field, value in writes.items():
            tracker.field("state.write", location, result, field, value, field=field, **details)
    return result


def replace_state(location, value):
    tracker = ACTIVE_STATE.get()
    if tracker is not None:
        tracker.remember(value, ("state",))
        tracker.emit("state.write", "state", source=location, path="state", value=value)
    return value
