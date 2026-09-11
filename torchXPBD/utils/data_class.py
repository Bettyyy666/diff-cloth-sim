"""Minimal geometry containers used by :mod:`torch_xpbd`.

``torch_xpbd.py`` was written against the ``dress-anyone2`` repo's
``utils.data_class``; this is the same (tiny) surface it actually touches --
``Mesh.vertices`` / ``Mesh.faces`` -- reproduced here so the simulator runs
standalone in this repo.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Mesh:
    """A triangle mesh: ``vertices`` ``(V, 3)`` float, ``faces`` ``(F, 3)`` int."""

    vertices: np.ndarray
    faces: np.ndarray

    def __post_init__(self) -> None:
        self.vertices = np.ascontiguousarray(self.vertices, dtype=np.float64)
        self.faces = np.ascontiguousarray(self.faces, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.vertices)
