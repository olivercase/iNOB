"""Convert Wavefront OBJ surface meshes to STL format.

This utility is used within the vagus nerve forward modelling pipeline to
standardise input surface geometries prior to volumetric mesh generation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import trimesh
except ImportError:
    sys.stderr.write(
        "Error: 'trimesh' is not installed. "
        "Install it with 'pip install trimesh' and try again.\n"
    )
    sys.exit(1)


def convert_obj_to_stl(obj_path: Path, stl_path: Path) -> Path:
    """Convert an OBJ surface mesh to STL format.

    Parameters
    ----------
    obj_path
        Path to the input Wavefront OBJ file.
    stl_path
        Destination path for the output STL file. The ``.stl`` suffix is
        appended automatically if absent.

    Returns
    -------
    Path
        The resolved path to the exported STL file.

    Raises
    ------
    FileNotFoundError
        If ``obj_path`` does not exist.
    ValueError
        If the input and output paths resolve to the same file.
    """
    obj_path = Path(obj_path)
    stl_path = Path(stl_path)

    if not obj_path.is_file():
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")

    if stl_path.suffix.lower() != ".stl":
        stl_path = stl_path.with_suffix(stl_path.suffix + ".stl")

    if obj_path.resolve() == stl_path.resolve():
        raise ValueError("Input and output paths must differ.")

    mesh = trimesh.load_mesh(obj_path)
    mesh.export(stl_path)
    return stl_path


def _parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Wavefront OBJ mesh to STL format."
    )
    parser.add_argument("obj_path", nargs="?", type=Path, help="Input OBJ file.")
    parser.add_argument("stl_path", nargs="?", type=Path, help="Output STL file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_arguments(argv)

    obj_path = args.obj_path or Path(input("Enter path to OBJ file: ").strip())
    stl_path = args.stl_path or Path(input("Enter path for output STL file: ").strip())

    try:
        output_path = convert_obj_to_stl(obj_path, stl_path)
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1

    print(f"Converted {obj_path} to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
