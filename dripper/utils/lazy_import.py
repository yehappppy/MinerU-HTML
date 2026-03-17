# ruff: noqa

import importlib
from typing import Any, Optional


class LazyImport:
    """
    Lazy proxy that behaves like the imported object.
    When wrapping a class, it can be used directly with isinstance().

    Usage:
        Paragraph = lazy_from("docx.text.paragraph", "Paragraph")
        isinstance(some_paragraph, Paragraph)  # Works!
    """

    def __init__(self, module_name: str, import_name: Optional[str] = None):
        self._module_name = module_name
        self._import_name = import_name
        self._loaded: Optional[Any] = None

    def _load(self) -> Any:
        if self._loaded is not None:
            return self._loaded

        mod = importlib.import_module(self._module_name)

        if self._import_name is None:
            target = mod
        else:
            try:
                target = getattr(mod, self._import_name)
            except AttributeError as exc:
                raise ImportError(
                    f"Cannot lazy-import {self._import_name!r} from {self._module_name!r}\n"
                    f"→ {exc.__class__.__name__}: {exc}"
                ) from None

        self._loaded = target
        return target

    @property
    def _class(self) -> type:
        """
        Returns the actual class if this LazyImport wraps a class.
        Use this for isinstance checks when direct isinstance() doesn't work.
        """
        loaded = self._load()
        if isinstance(loaded, type):
            return loaded
        raise TypeError(f"LazyImport {self._module_name}.{self._import_name} is not a class")

    def __getattr__(self, name: str) -> Any:
        real = self._load()
        return getattr(real, name)

    def __class__(self):
        """
        Override __class__ so type(obj) returns the actual type, not LazyImport.
        This is critical for type introspection to work correctly.
        """
        loaded = self._load()
        return loaded.__class__

    def __call__(self, *args, **kwargs) -> Any:
        real = self._load()
        if not callable(real):
            raise TypeError(
                f"Cannot call {self._module_name}.{self._import_name or ''!r} — "
                f"it is not callable (got {type(real).__name__})"
            )
        return real(*args, **kwargs)

    def __repr__(self) -> str:
        if self._loaded is None:
            name = self._import_name or self._module_name
            return f"<LazyImport {name!r} from {self._module_name!r} (not loaded)>"
        return repr(self._loaded)

    # Optional: better dir() / introspection before loading
    def __dir__(self) -> list:
        if self._loaded is not None:
            return dir(self._loaded)

        try:
            mod = importlib.import_module(self._module_name)
            if self._import_name is None:
                return dir(mod)
            return dir(getattr(mod, self._import_name))
        except Exception:
            return ['__call__', '__getattr__', '_load']

    # Support isinstance by returning True for type checks against the loaded class
    def __instancecheck__(self, instance) -> bool:
        """Enable isinstance(instance, LazyImport) to work with the loaded class."""
        loaded = self._load()
        if isinstance(loaded, type):
            return isinstance(instance, loaded)
        return isinstance(instance, type(loaded))

    def __subclasscheck__(self, subclass) -> bool:
        """Enable issubclass() to work with the loaded class."""
        loaded = self._load()
        if isinstance(loaded, type):
            return issubclass(subclass, loaded)
        return False


def lazy_module(module_path: str) -> Any:
    """For `import torch` / `import numpy as np` style"""
    return LazyImport(module_path)


def lazy_from(module_path: str, name: str) -> Any:
    """
    For `from module import name` or `from module import name as alias`

    Examples:
        md = lazy_from("markdownify", "markdownify")
        clean = lazy_from("bleach", "clean")
        List = lazy_from("typing", "List")
    """
    return LazyImport(module_path, import_name=name)


def unwrap(obj: Any) -> Any:
    """
    Unwrap a LazyImport to get the actual loaded object.

    Args:
        obj: A LazyImport instance or any other object

    Returns:
        The actual loaded object if obj is a LazyImport, otherwise obj unchanged
    """
    if isinstance(obj, LazyImport):
        return obj._load()
    return obj
