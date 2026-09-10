"""Multi-projection geometry visualisation (matplotlib).

Renders a 4-panel PNG (lateral / posterior / axial projections + 3-D view)
of the geometry HDF5 produced by :mod:`inob.geometry.builder`. Optionally
overlays the sensor array on each panel.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from inob.config import Config
from inob.io.hdf5 import load_geometry, load_sensors

logger = logging.getLogger(__name__)

MESH_STYLES: dict[str, dict] = {
    "mesh_skin": dict(colour="#E8C5A0", alpha=0.10, label="Skin"),
    "mesh_bone": dict(colour="#DDCC77", alpha=0.45, label="Bone"),
    "mesh_vagus_left": dict(colour="#44BB99", alpha=0.95, label="Left vagus nerve"),
    "mesh_vagus_right": dict(colour="#CC6677", alpha=0.95, label="Right vagus nerve"),
}
MAX_TRIS_3D = 12_000
MAX_TRIS_PROJ = 60_000


def _subsample_faces(faces: np.ndarray, max_tris: int, rng: np.random.Generator) -> np.ndarray:
    if len(faces) <= max_tris:
        return faces
    idx = rng.choice(len(faces), max_tris, replace=False)
    return faces[idx]


def _draw_projection(ax, meshes, xi, yi, xlabel, ylabel, title, rng):
    for key, (vertices, faces) in meshes.items():
        style = MESH_STYLES.get(key, dict(colour="grey", alpha=0.3))
        sub = _subsample_faces(faces, MAX_TRIS_PROJ, rng)
        pts = vertices[sub].reshape(-1, 3)
        ax.scatter(
            pts[:, xi],
            pts[:, yi],
            s=0.3,
            c=style["colour"],
            alpha=min(style["alpha"] + 0.1, 0.9),
            linewidths=0,
            rasterized=True,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.4)


def _draw_3d(ax, meshes, rng):
    for key, (vertices, faces) in meshes.items():
        style = MESH_STYLES.get(key, dict(colour="grey", alpha=0.3))
        sub = _subsample_faces(faces, MAX_TRIS_3D, rng)
        coll = Poly3DCollection(
            vertices[sub],
            alpha=style["alpha"],
            facecolor=style["colour"],
            edgecolor="none",
        )
        ax.add_collection3d(coll)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("3-D perspective")
    ax.view_init(elev=15, azim=35)


def render_geometry(
    cfg: Config,
    *,
    dpi: int = 150,
    show: bool = False,
    with_sensors: bool = True,
) -> Path:
    """Render the 4-panel geometry overview to ``cfg.outputs.geometry_png``."""
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
        }
    )
    rng = np.random.default_rng(cfg.reproducibility.seed)

    mat_path = cfg.outputs.geometry_mat
    if not mat_path.exists():
        raise FileNotFoundError(f"{mat_path} not found. Run build_geom first.")
    geom = load_geometry(mat_path)
    meshes: dict[str, tuple[np.ndarray, np.ndarray]] = {
        name: (c.vertices, c.faces) for name, c in geom.compartments.items()
    }
    all_pts = np.vstack([v for v, _ in meshes.values()])
    lo = all_pts.min(0)
    hi = all_pts.max(0)
    span = hi - lo
    lo, hi = lo - 0.05 * span, hi + 0.05 * span
    xlim, ylim, zlim = (lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2])

    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(f"iNOB Geometry — {mat_path.name}", fontsize=11, y=0.98)
    gs = fig.add_gridspec(
        2, 2, hspace=0.30, wspace=0.25, left=0.07, right=0.96, top=0.94, bottom=0.06
    )
    ax_lat = fig.add_subplot(gs[0, 0])
    ax_post = fig.add_subplot(gs[0, 1])
    ax_ax = fig.add_subplot(gs[1, 0])
    ax_3d = fig.add_subplot(gs[1, 1], projection="3d")

    _draw_projection(ax_lat, meshes, 0, 1, "X (mm)", "Y (mm)", "Lateral (X – Y)", rng)
    _draw_projection(ax_post, meshes, 2, 1, "Z (mm)", "Y (mm)", "Posterior (Z – Y)", rng)
    _draw_projection(ax_ax, meshes, 0, 2, "X (mm)", "Z (mm)", "Axial (X – Z)", rng)
    _draw_3d(ax_3d, meshes, rng)

    if with_sensors and cfg.outputs.sensors_mat.exists():
        sensors = load_sensors(cfg.outputs.sensors_mat)
        n = len(sensors.coilpos) // 3
        positions = sensors.coilpos[:n]
        normals = sensors.coilori[:n]
        for ax, (xi, yi) in ((ax_lat, (0, 1)), (ax_post, (2, 1)), (ax_ax, (0, 2))):
            ax.scatter(
                positions[:, xi],
                positions[:, yi],
                s=4,
                c="#332288",
                alpha=0.85,
                linewidths=0,
                rasterized=True,
            )
        ax_3d.scatter(
            positions[:, 0], positions[:, 1], positions[:, 2], s=4, c="#332288", depthshade=False
        )
        step = max(1, len(positions) // 200)
        P = positions[::step]
        N = normals[::step] * 25.0
        ax_3d.quiver(
            P[:, 0],
            P[:, 1],
            P[:, 2],
            N[:, 0],
            N[:, 1],
            N[:, 2],
            color="#882255",
            linewidth=0.5,
            arrow_length_ratio=0.0,
        )

    ax_lat.set_xlim(xlim)
    ax_lat.set_ylim(ylim)
    ax_post.set_xlim(zlim)
    ax_post.set_ylim(ylim)
    ax_ax.set_xlim(xlim)
    ax_ax.set_ylim(zlim)
    ax_3d.set_xlim(xlim)
    ax_3d.set_ylim(ylim)
    ax_3d.set_zlim(zlim)

    import matplotlib.patches as mpatches

    patches = [
        mpatches.Patch(
            color=MESH_STYLES[k]["colour"],
            alpha=max(MESH_STYLES[k]["alpha"], 0.5),
            label=MESH_STYLES[k]["label"],
        )
        for k in meshes
        if k in MESH_STYLES
    ]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=min(len(patches), 5),
        framealpha=0.9,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.01),
    )

    out = cfg.outputs.geometry_png
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight")
    logger.info("[saved] %s", out)
    if show:
        plt.show()
    plt.close(fig)
    return out
