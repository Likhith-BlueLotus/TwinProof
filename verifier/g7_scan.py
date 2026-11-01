"""G7 AST guard: refuse code that writes simulator state, edits physics options/goals, or reaches outside.

A best-effort lint for the commit hook and for agent-proposed diffs, not a sandbox: LLM output is never
executed (controllers are parametric). CLI: `python -m verifier.g7_scan [files...]` exits 2, reasons on stderr.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUARDED_DIRS = ("physics", "verifier", "agents")
RESET_MODULE = "physics/reset.py"
STATE_ATTRS = {"qpos", "qvel", "time"}
OPTION_ATTRS = {"disableflags", "gravity", "timestep"}
BANNED_CALLS = {
    "eval", "exec", "compile", "__import__", "setattr", "getattr", "delattr", "globals", "locals", "vars",
    "time.sleep", "os.system", "os.popen", "os.kill", "os.fork",
}
BANNED_CALL_PREFIXES = ("os.exec", "os.spawn")
BANNED_MODULES = {
    "socket", "requests", "httpx", "urllib", "subprocess", "ctypes", "multiprocessing", "importlib",
    "shutil", "http", "pickle", "builtins",
}
STATE_SETTERS = {"mj_resetData", "mj_resetDataKeyframe", "mj_resetDataDebug", "mj_setState"}
MUTATING_METHODS = {"fill", "put", "itemset", "resize", "sort", "partition", "setfield", "__setitem__"}
ARG_MUTATORS = {"copyto", "put", "place", "putmask", "fill_diagonal"}  # write into their first argument


def _dotted(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Subscript):
        return _dotted(node.value)
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _flatten(target: ast.AST):
    if isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            yield from _flatten(elt)
    elif isinstance(target, ast.Starred):
        yield from _flatten(target.value)
    else:
        yield target


def _targets(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            raw = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.For, ast.AsyncFor, ast.NamedExpr)):
            raw = [node.target]
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            raw = [i.optional_vars for i in node.items if i.optional_vars is not None]
        else:
            continue
        for t in raw:
            for leaf in _flatten(t):
                yield node, leaf


def _is_state(name: str) -> bool:
    parts = name.split(".")
    return len(parts) > 1 and (parts[-1] in STATE_ATTRS or "opt" in parts[1:])


def scan_claim(code: dict[str, str]) -> list[str]:
    """Agent-submitted code is never the reset module, whatever path it claims to be."""
    return [v for path, src in code.items() for v in scan_source(src, f"claim/{path}")]


def scan_source(source: str, path: str) -> list[str]:
    """Violations as 'path:line: reason'; empty means the file passes."""
    path = Path(path).as_posix()
    exempt = path == RESET_MODULE
    try:
        tree = ast.parse(source)
    except SyntaxError as err:
        return [f"{path}:{err.lineno}: does not parse"]
    found: list[str] = []
    aliases: dict[str, str] = {}  # local name -> resolved dotted name

    def resolve(node: ast.AST) -> str:
        name = _dotted(node)
        head, _, rest = name.partition(".")
        head = aliases.get(head, head)
        return f"{head}.{rest}" if rest else head

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                aliases[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
                if a.name.split(".")[0] in BANNED_MODULES:
                    found.append(f"{path}:{node.lineno}: banned import {a.name}")
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                full = f"{node.module}.{a.name}" if node.module else a.name
                aliases[a.asname or a.name] = full
                if full.split(".")[0] in BANNED_MODULES or full in BANNED_CALLS:
                    found.append(f"{path}:{node.lineno}: banned import {full}")

    for node, target in _targets(tree):
        name = resolve(target)
        leaf = name.rsplit(".", 1)[-1]
        if _is_state(name) and not exempt:
            found.append(f"{path}:{node.lineno}: writes simulator state or options {name}")
        if leaf.startswith("GOAL"):
            found.append(f"{path}:{node.lineno}: reassigns goal constant {leaf}")
        value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)) else None
        if isinstance(target, ast.Name) and value is not None:
            bound = resolve(value)
            if _is_state(bound) and not exempt:
                found.append(f"{path}:{node.lineno}: aliases simulator state {bound} as {target.id}")
            if bound in BANNED_CALLS:
                found.append(f"{path}:{node.lineno}: aliases banned callable {bound}")

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = resolve(node.func)
        if name in BANNED_CALLS or name.startswith(BANNED_CALL_PREFIXES) or name.split(".")[0] in BANNED_MODULES:
            found.append(f"{path}:{node.lineno}: banned call {name}()")
        if exempt:
            continue
        if name.rsplit(".", 1)[-1] in STATE_SETTERS:
            found.append(f"{path}:{node.lineno}: state setter {name}() outside {RESET_MODULE}")
        if isinstance(node.func, ast.Attribute) and node.func.attr in MUTATING_METHODS and _is_state(resolve(node.func.value)):
            found.append(f"{path}:{node.lineno}: mutates simulator state via {name}()")
        writes_arg = name.rsplit(".", 1)[-1] in ARG_MUTATORS and node.args and _is_state(resolve(node.args[0]))
        writes_out = any(k.arg == "out" and _is_state(resolve(k.value)) for k in node.keywords)
        if writes_arg or writes_out:
            found.append(f"{path}:{node.lineno}: writes simulator state via {name}()")
    return found


def scan_paths(paths: list[str]) -> list[str]:
    found = []
    for p in paths:
        full = Path(p).resolve()
        try:
            rel = full.relative_to(ROOT)
        except ValueError:
            continue  # outside the repo
        if full.suffix == ".py" and rel.parts[0] in GUARDED_DIRS:
            found += scan_source(full.read_text(), rel.as_posix())
    return found


if __name__ == "__main__":
    targets = sys.argv[1:] or [str(p) for d in GUARDED_DIRS for p in (ROOT / d).rglob("*.py")]
    violations = scan_paths(targets)
    for v in violations:
        print(v, file=sys.stderr)
    sys.exit(2 if violations else 0)
