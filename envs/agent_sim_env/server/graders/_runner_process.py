"""Child-process entry point for generated grader functions."""

from __future__ import annotations

import builtins
import json
import sys
from typing import Any

try:  # pragma: no cover - unavailable on Windows
    import resource
except ImportError:  # pragma: no cover
    resource = None


_ALLOWED_IMPORTS = frozenset({"sqlite3", "json", "re", "math", "collections"})
_SAFE_BUILTINS = frozenset(
    {
        "True",
        "False",
        "None",
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "Exception",
        "float",
        "int",
        "isinstance",
        "len",
        "list",
        "max",
        "min",
        "range",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    }
)


def _safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
    if name.split(".", 1)[0] not in _ALLOWED_IMPORTS:
        raise ImportError(f"Import of {name!r} is not permitted")
    return builtins.__import__(name, *args, **kwargs)


def _apply_limits() -> None:
    if resource is None:
        return
    limits = (
        (resource.RLIMIT_CPU, 5),
        (resource.RLIMIT_AS, 256 * 1024 * 1024),
        (resource.RLIMIT_FSIZE, 4 * 1024 * 1024),
    )
    for limit, value in limits:
        try:
            resource.setrlimit(limit, (value, value))
        except (OSError, ValueError, resource.error):
            continue


def main() -> int:
    _apply_limits()
    try:
        payload = json.loads(sys.stdin.read())
        code = payload["code"]
        function_name = payload["function_name"]
        arguments = payload["arguments"]
        safe_builtins = {
            name: getattr(builtins, name)
            for name in _SAFE_BUILTINS
            if hasattr(builtins, name)
        }
        safe_builtins["__import__"] = _safe_import
        namespace: dict[str, Any] = {
            "__builtins__": safe_builtins,
            "__name__": "__grader__",
        }
        exec(code, namespace)  # noqa: S102 - isolated generated code
        grade = namespace.get(function_name)
        if not callable(grade):
            raise ValueError(f"Missing callable {function_name!r}")
        result = grade(**arguments)
        if not isinstance(result, dict):
            raise ValueError("grader must return a dictionary")
        print(json.dumps({"status": "ok", **result}))
    except Exception as error:  # pylint: disable=broad-except
        print(json.dumps({"status": "error", "reason": str(error)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
