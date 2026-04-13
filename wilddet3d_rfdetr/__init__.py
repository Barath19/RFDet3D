"""RFDet3D: RF-DETR + 3D Detection Head.

Commercially-viable monocular 3D object detection using
RF-DETR (Apache 2.0) as the 2D detector with WildDet3D's
3D detection head and LingBot-Depth geometry backend.
"""

from wilddet3d_rfdetr._setup_paths import *  # noqa: F401, F403

from wilddet3d_rfdetr.data_types import RFDet3DInput, RFDet3DOut
from wilddet3d.data_types import Det3DOut
from wilddet3d_rfdetr.model import RFDet3D

__all__ = [
    "RFDet3D",
    "RFDet3DInput",
    "RFDet3DOut",
    "Det3DOut",
]
