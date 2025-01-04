import numpy as np
from pathlib import Path

def load_stl(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, 'rb') as f:
        f.read(80)  # Skip header
        num_triangles = int.from_bytes(f.read(4), 'little')
        bytedata = np.frombuffer(f.read(num_triangles * (12 * 4 + 2)), np.uint8).reshape(num_triangles, 12 * 4 + 2)
        floatdata = bytedata[:, :-2].view(np.float32).reshape(num_triangles, 4, 3)
        normals = floatdata[:, :1].repeat(3, axis=1)
        vertices = floatdata[:, 1:]
        return vertices.copy(), normals.copy()

def load_obj(path: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices, normals, faces = [], [], []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append([float(x) for x in line.strip().split()[1:4]])
            elif line.startswith('vn '):
                normals.append([float(x) for x in line.strip().split()[1:4]])
            elif line.startswith('f '):
                faces.append([int(vertex.split('/')[0]) - 1 for vertex in line.strip().split()[1:]])  # OBJ indices are 1-based
    return np.array(vertices, dtype=np.float32), np.array(normals, dtype=np.float32), np.array(faces, dtype=np.uint32)
