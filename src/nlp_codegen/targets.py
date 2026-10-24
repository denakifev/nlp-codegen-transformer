import ast
from decimal import Decimal, InvalidOperation
from functools import lru_cache
import math
import re


OPS = ("Add", "Sub", "Mult", "Div", "FloorDiv", "Mod", "Pow", "UAdd", "USub")
SIZES = (5,) * len(OPS) + (10, 8, 8)
NUMBER = re.compile(r"(?<![\w.])[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?![\w.])")
BINARY = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
UNARY = (ast.UAdd, ast.USub)


class Unsupported(ValueError):
    pass


def _parse(code):
    if not isinstance(code, str) or len(code) > 65536:
        raise Unsupported("size")
    tree = ast.parse(code)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise Unsupported("module")
    function = tree.body[0]
    args = function.args
    if (function.name != "solve" or function.decorator_list or function.returns is not None
            or args.args or args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg
            or args.defaults or args.kw_defaults or getattr(function, "type_params", ())):
        raise Unsupported("signature")
    nodes, environment = [], {}

    def expression(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if len(repr(node.value)) > 100 or not math.isfinite(float(node.value)):
                raise Unsupported("literal")
            value = ("const", node.value)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in environment:
                raise Unsupported("name")
            return environment[node.id]
        elif isinstance(node, ast.BinOp) and isinstance(node.op, BINARY):
            value = ("binary", type(node.op).__name__, expression(node.left), expression(node.right))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, UNARY):
            value = ("unary", type(node.op).__name__, expression(node.operand))
        else:
            raise Unsupported("expression")
        if len(nodes) >= 4096:
            raise Unsupported("nodes")
        nodes.append(value)
        return len(nodes) - 1

    root = None
    for position, statement in enumerate(function.body):
        if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name) and statement.type_comment is None):
            value = expression(statement.value)
            environment[statement.targets[0].id] = value
        elif isinstance(statement, ast.Return) and position == len(function.body) - 1 and statement.value is not None:
            root = expression(statement.value)
        else:
            raise Unsupported("statement")
    if root is None:
        raise Unsupported("return")
    return nodes, root, tree


def _numeric_key(value):
    try:
        result = Decimal(str(value).replace(",", ""))
    except (InvalidOperation, ValueError):
        return None
    return result.normalize() if result.is_finite() else None


def _offset(code, line, column):
    lines = code.splitlines(keepends=True)
    prefix = sum(map(len, lines[:line - 1]))
    character_column = len(lines[line - 1].encode("utf-8")[:column].decode("utf-8"))
    return prefix + character_column


@lru_cache(maxsize=8192)
def _program_labels(code):
    nodes, root, tree = _parse(code)
    live, pending = set(), [root]
    while pending:
        index = pending.pop()
        if index in live:
            continue
        live.add(index)
        if nodes[index][0] != "const":
            pending.extend(nodes[index][2:])
    counts = [min(4, sum(nodes[index][0] != "const" and nodes[index][1] == operation for index in live))
              for operation in OPS]
    depths = []
    for node in nodes:
        depths.append(0 if node[0] == "const" else 1 + max(depths[index] for index in node[2:]))
    kind = 0 if nodes[root][0] == "const" else 1 + OPS.index(nodes[root][1])
    structure = tuple(counts + [kind, min(7, depths[root]), min(7, sum(nodes[index][0] == "const" for index in live))])
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    literals = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or type(node.value) not in (int, float):
            continue
        value, current = node.value, node
        while isinstance(parents.get(current), ast.UnaryOp) and isinstance(parents[current].op, UNARY):
            current = parents[current]
            if isinstance(current.op, ast.USub):
                value = -value
        start = _offset(code, node.lineno, node.col_offset)
        end = _offset(code, node.end_lineno, node.end_col_offset)
        literals.append((start, end, _numeric_key(value)))
    return structure, tuple(literals)


def derive_auxiliary_labels(source_ids, target_ids, tokenizer):
    structure = [-100] * len(SIZES)
    grounding = [-100.0] * len(target_ids)
    if not source_ids or not target_ids or source_ids[-1] != 2 or target_ids[-1] != 2:
        return structure, grounding
    try:
        question = tokenizer.decode(list(source_ids[:-1]), skip_special_tokens=False)
        code = tokenizer.decode(list(target_ids[:-1]), skip_special_tokens=False)
        source_encoding = tokenizer.encode(question, add_special_tokens=False)
        target_encoding = tokenizer.encode(code, add_special_tokens=False)
        if source_encoding.ids != list(source_ids[:-1]) or target_encoding.ids != list(target_ids[:-1]):
            return structure, grounding
        labels, literals = _program_labels(code)
        question_values = {_numeric_key(match.group()) for match in NUMBER.finditer(question)} - {None}
        structure = list(labels)
        candidates = [set() for _ in target_encoding.ids]
        for start, end, value in literals:
            label = float(value in question_values)
            for index, (left, right) in enumerate(target_encoding.offsets):
                if left < end and right > start:
                    candidates[index].add(label)
        for index, values in enumerate(candidates):
            if len(values) == 1:
                grounding[index] = next(iter(values))
        return structure, grounding
    except (Unsupported, SyntaxError, TypeError, ValueError, RecursionError, OverflowError):
        return [-100] * len(SIZES), [-100.0] * len(target_ids)
