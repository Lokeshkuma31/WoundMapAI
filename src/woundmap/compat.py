"""Python 3.14 and platform compatibility shims for PyTorch."""
import sys
import types

def ensure_dynamo_compat():
    """Fix for Python 3.14 PyTorch pre-release circular import bug in torch._dynamo."""
    if "torch._dynamo" not in sys.modules:
        try:
            import torch._dynamo  # noqa: F401
        except ImportError:
            dummy_dynamo = types.ModuleType("torch._dynamo")
            dummy_dynamo.is_compiling = lambda: False
            dummy_dynamo.disable = lambda fn, *args, **kwargs: fn
            dummy_dynamo.graph_break = lambda *args, **kwargs: None
            sys.modules["torch._dynamo"] = dummy_dynamo

ensure_dynamo_compat()
