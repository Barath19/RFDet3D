"""Path and import setup for wilddet3d components.

Stubs out sam3 so we can import wilddet3d submodules (head, depth, ops)
without requiring SAM3's CUDA-only dependencies (triton, etc.).
"""

import sys
import types
from pathlib import Path

_root = Path(__file__).parent.parent
_wilddet3d_root = _root / "third_party" / "WildDet3D"
_lingbot_root = _wilddet3d_root / "third_party" / "lingbot_depth"

# Add paths
for p in [str(_wilddet3d_root), str(_lingbot_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)


def _stub_sam3():
    """Insert a fake sam3 package into sys.modules.

    This prevents wilddet3d's __init__.py from crashing when it tries
    to import sam3 (which requires triton/CUDA). We never call SAM3 code
    in the RF-DETR pipeline, so the stub is safe.
    """
    if "sam3" in sys.modules:
        return

    class _StubModule(types.ModuleType):
        """A module stub that auto-creates sub-attributes on access."""

        def __init__(self, name):
            super().__init__(name)
            self.__file__ = f"<stub:{name}>"
            self.__path__ = []
            self.__package__ = name
            self.__loader__ = None
            self.__spec__ = None

        def __getattr__(self, name):
            # Return a new stub for sub-module access
            full_name = f"{self.__name__}.{name}"
            if full_name not in sys.modules:
                sub = _StubModule(full_name)
                sys.modules[full_name] = sub
            return sys.modules[full_name]

    # Pre-register the stub hierarchy
    for mod_name in [
        "sam3",
        "sam3.model",
        "sam3.model.sam3_image",
        "sam3.model.geometry_encoders",
        "sam3.model.box_ops",
        "sam3.model.data_misc",
        "sam3.model_builder",
        "sam3.train",
        "sam3.train.matcher",
        "sam3.train.loss",
        "sam3.train.loss.loss_fns",
    ]:
        sys.modules[mod_name] = _StubModule(mod_name)


# Stub sam3 before any wilddet3d imports
_stub_sam3()
