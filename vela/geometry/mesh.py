import numpy as np
from pathlib import Path

# Helper functions

def compute_normals(vertices: np.ndarray) -> np.ndarray:
    v0, v1, v2 = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    face_normals = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-24
    face_normals /= norm
    return np.repeat(face_normals, 3, axis=0)

def join_faces(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices = vertices.reshape(-1, 3)
    unique_vertices, inverse_indices = np.unique(vertices, axis=0, return_inverse=True)
    indices = inverse_indices.reshape(-1, 3)
    return unique_vertices, indices

def split_faces(faces: np.ndarray, vertices: np.ndarray, normals: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
    return vertices[faces], normals[faces] if normals is not None else None


# Loaders

def load_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, 'rb') as f:
        f.read(80)  # Skip header
        num_triangles = int.from_bytes(f.read(4), 'little')
        bytedata = np.frombuffer(f.read(num_triangles * (12 * 4 + 2)), np.uint8).reshape(num_triangles, 12 * 4 + 2)
        floatdata = bytedata[:, :-2].view(np.float32).reshape(num_triangles, 4, 3)
        normals = floatdata[:, :1].repeat(3, axis=1)
        vertices = floatdata[:, 1:]
        return vertices, normals

def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices_list, normals_list, faces_list = [], [], []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices_list.append([float(x) for x in line.strip().split()[1:4]])
            elif line.startswith('vn '):
                normals_list.append([float(x) for x in line.strip().split()[1:4]])
            elif line.startswith('f '):
                faces_list.append([int(vertex.split('/')[0]) - 1 for vertex in line.strip().split()[1:]])  # OBJ indices are 1-based

    faces = np.array(faces_list, dtype=np.uint32)
    vertices = np.array(vertices_list, dtype=np.float32)
    normals = np.array(normals_list, dtype=np.float32) if normals_list else None
    vertices, normals = split_faces(faces, vertices, normals)
    if normals is None:
        normals = compute_normals(vertices)
    return vertices, normals

def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix == '.stl':
        return load_stl(path)
    elif path.suffix == '.obj':
        return load_obj(path)
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")
