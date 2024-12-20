import numpy as np
from pathlib import Path

def parse_stl(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    normals = []

    with open(path, 'rb') as f:
        f.read(80)  # Skip header

        num_triangles = int.from_bytes(f.read(4), 'little')
        for _ in range(num_triangles):
            # Read normal vector (3 floats)
            nx = float(np.frombuffer(f.read(4), np.float32)[0])
            ny = float(np.frombuffer(f.read(4), np.float32)[0])
            nz = float(np.frombuffer(f.read(4), np.float32)[0])
            normals.append([nx, ny, nz])

            # Read 3 vertices (9 floats)
            tri_verts = []
            for _ in range(3):
                x = float(np.frombuffer(f.read(4), np.float32)[0])
                y = float(np.frombuffer(f.read(4), np.float32)[0])
                z = float(np.frombuffer(f.read(4), np.float32)[0])
                tri_verts.append([x, y, z])
            vertices.append(tri_verts)

            # Skip attribute byte count
            f.read(2)

    return np.array(vertices, dtype=np.float32), np.array(normals, dtype=np.float32)
