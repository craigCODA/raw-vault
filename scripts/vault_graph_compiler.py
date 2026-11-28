"""Lower trusted vault-body-graph/1 instruction records to closed Python AST.

A function is assembled from addressed statement edges. Expressions contain
only literal, local, linked body, and approved syntax nodes. No graph field is
parsed as Python source. Training data is never passed to this compiler.
"""

import ast
import keyword
from types import CodeType, FunctionType


SYNTAX = {
    name: getattr(ast, name) for name in """
    Add And Assign Attribute AugAssign BinOp BitOr BoolOp Break Call Compare
    Continue Del Delete Dict DictComp Div Eq ExceptHandler Expr FloorDiv For
    FormattedValue GeneratorExp Gt GtE If IfExp In Is IsNot JoinedStr Lambda
    List ListComp Load Lt LtE Mod Mult Not NotEq NotIn Or Raise Return Set
    SetComp Slice Starred Store Sub Subscript Try Tuple USub UnaryOp While
    Yield YieldFrom arg arguments comprehension keyword
    """.split()
}
ATTRIBUTES = set("""
    I add append bytes_read casefold catalog copy denominator dumps encode
    evaluate extend findall format format_map fromisoformat fromkeys fullmatch
    get get_traced_memory group groupdict hexdigest inputs intersection
    is_tracing isdecimal isoformat issubset items join keys loads numerator
    operand operations_run pop popleft read_asset remove reverse root rstrip run
    setdefault sha256 split start startswith stop strip trace union update values
""".split())
# Missing optional fields have the same meaning on Python 3.12 and newer.
SEQUENCES = set("""
    args body comparators defaults elts finalbody generators handlers ifs
    keys keywords kw_defaults kwonlyargs ops orelse posonlyargs targets values
""".split())


def identifier(name):
    if (not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name)
            or name.startswith("_") and name not in {"_", "_parameters", "_context"}):
        raise ValueError(f"Invalid graph identifier {name!r}")
    return name


def host_call(name, *args):
    return ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=list(args), keywords=[])


def marker(kind, address):
    return ast.Expr(host_call("_graph_mark", ast.Constant(kind), ast.Constant(address)))


class StateInstructions(ast.NodeTransformer):
    """Measure operations at their original evaluation sites, including comprehensions."""

    def __init__(self, binding_lookup):
        self.binding_lookup = binding_lookup

    def visit_Subscript(self, node):
        self.generic_visit(node)
        if not self.binding_lookup and isinstance(node.ctx, ast.Load) and not isinstance(node.slice, ast.Slice):
            return host_call("_read_item", ast.Constant(node.vault_location), node.value, node.slice)
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript):
            target = node.targets[0]
            if not isinstance(target.slice, ast.Slice):
                # Python evaluates the right-hand side before the store target.
                return ast.Expr(host_call("_write_item", ast.Constant(node.vault_location),
                                          node.value, target.value, target.slice))
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        target = node.func
        if isinstance(target, ast.Attribute) and target.attr in {"get", "copy", "setdefault", "update"}:
            node.args[:0] = [ast.Constant(node.vault_location), target]
            node.func = ast.Name(id="_mapping_call", ctx=ast.Load())
        elif isinstance(target, ast.Name) and target.id in {"dict", "deepcopy"} and node.args:
            node.args.insert(0, target)
            node.func = ast.Name(id="_copy_mapping", ctx=ast.Load())
        return node


class BodyCompiler:
    def __init__(self, address, body, symbols):
        self.address = address
        self.body = body
        self.symbols = symbols
        self.active = set()

    def instructions(self, labels):
        block = []
        for label in labels:
            if label in self.active or label not in self.body["nodes"]:
                raise ValueError(f"Cyclic or missing instruction {self.address}/{label}")
            location = f"{self.address}/{label}"
            self.active.add(label)
            node = self.decode(self.body["nodes"][label], location)
            self.active.remove(label)
            if not isinstance(node, ast.stmt):
                raise ValueError(f"Expected a statement at {location}")
            block.extend((marker("graph.step", location), node))
        return block

    def decode(self, value, location):
        if isinstance(value, list):
            if value and all(isinstance(item, dict) and set(item) == {"step"} for item in value):
                return self.instructions(item["step"] for item in value)
            return [self.decode(item, location) for item in value]
        if not isinstance(value, dict):
            return value
        if set(value) == {"const"}:
            return ast.Constant(value["const"])
        if set(value) == {"body"}:
            if value["body"] not in self.symbols:
                raise ValueError(f"Unresolved body edge at {location}: {value['body']}")
            return ast.Name(id=self.symbols[value["body"]], ctx=ast.Load())
        for tag, ctx in (("name", ast.Load), ("store", ast.Store), ("del", ast.Del)):
            if set(value) == {tag}:
                return ast.Name(id=identifier(value[tag]), ctx=ctx())
        kind = value.get("op")
        if kind not in SYNTAX:
            raise ValueError(f"Unknown graph instruction {kind!r} at {location}")
        fields = {}
        cls = SYNTAX[kind]
        for key, item in value.items():
            if key == "op":
                continue
            field = "op" if key == "operator" else key
            if field not in cls._fields:
                raise ValueError(f"Unknown field {key!r} at {location}")
            fields[field] = self.decode(item, location)
        for field in cls._fields:
            if field not in fields:
                if field == "ctx":
                    fields[field] = ast.Load()
                elif field in SEQUENCES or kind == "Call" and field == "args":
                    fields[field] = []
                else:
                    fields[field] = None
        if kind == "Attribute" and fields["attr"] not in ATTRIBUTES:
            raise ValueError(f"Unknown graph attribute at {location}: {fields['attr']!r}")
        if kind == "arg":
            identifier(fields["arg"])
        if kind == "keyword" and fields["arg"] is not None:
            identifier(fields["arg"])
        if kind == "Call" and isinstance(fields["func"], ast.Name):
            if fields["func"].id in {"record_read", "record_cursor", "replace_state", "state_patch"}:
                fields["args"].insert(0, ast.Constant(location))
        node = cls(**fields)
        node.vault_location = location
        return node

    def compile(self, namespace):
        parameters = [ast.arg(arg=identifier(p), annotation=None) for p in self.body["parameters"]]
        arguments = ast.arguments(posonlyargs=[], args=parameters, kwonlyargs=[], kw_defaults=[],
                                  defaults=[], vararg=None, kwarg=None)
        steps = self.instructions(self.body["entry"])
        statements = [marker("graph.call", self.address), *steps]
        function = ast.FunctionDef(name=self.symbols[self.address], args=arguments, body=statements,
                                   decorator_list=[], returns=None, type_comment=None, type_params=[])
        self.validate_calls(function, namespace)
        # get_value records the complete resolved path through record_read.
        StateInstructions(self.address == "body/vault_execution/get_value").visit(function)
        for call in (node for node in ast.walk(function) if isinstance(node, ast.Call)):
            target = call.func
            numeric = isinstance(target, ast.Name) and target.id in {"compile_value", "compile_equation", "Fraction"}
            if numeric or isinstance(target, ast.Attribute) and target.attr == "evaluate":
                call.args.insert(0, target)
                call.func = ast.Name(id="_arithmetic_call", ctx=ast.Load())
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        try:
            code = compile(module, self.address, "exec", dont_inherit=True)
        except (ValueError, TypeError, SyntaxError) as error:
            raise ValueError(f"Invalid instruction in {self.address}: {error}") from error
        function_code, = (item for item in code.co_consts if isinstance(item, CodeType))
        defaults = tuple(self.body.get("defaults", []))
        return FunctionType(function_code, namespace, self.address, defaults)

    def validate_calls(self, function, namespace):
        nodes = list(ast.walk(function))
        local = {node.arg for node in nodes if isinstance(node, ast.arg)}
        local.update(node.id for node in nodes if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store))
        allowed = (set(self.symbols.values()) | {name for name, obj in namespace.items() if callable(obj)}) - local
        for node in nodes:
            if not isinstance(node, ast.Call):
                continue
            target = node.func
            if isinstance(target, ast.Name) and target.id in allowed:
                if target.id == "type" and (len(node.args) != 1 or node.keywords):
                    raise ValueError(f"Only unary type inspection is available in {self.address}")
                continue
            if isinstance(target, ast.Attribute):
                continue  # Its attribute was checked when decoding the expression.
            if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                    and target.value.id in {"COMPARISONS", "COMMON_COMPARISONS"} - local):
                continue
            raise ValueError(f"Unbound or indirect call target in {self.address}")
