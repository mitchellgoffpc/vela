import numpy as np
from pathlib import Path

# Helper functions

def compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    face_normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    norm = np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-24
    return face_normals / norm

def compute_vertex_normals(face_normals: np.ndarray, faces: np.ndarray):
    vertex_normals = np.zeros((len(face_normals), 3), dtype=np.float32)
    for i, face in enumerate(faces):
        for j in range(3):
            vertex_normals[face[j]] += face_normals[i]
    norm = np.linalg.norm(vertex_normals, axis=1, keepdims=True) + 1e-24
    return vertex_normals / norm

def join_faces(vertices: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = vertices.reshape(-1, 3)
    normals = normals.reshape(-1, 3)
    combined = np.hstack((vertices, normals))
    unique_combined, inverse_indices = np.unique(combined, axis=0, return_inverse=True)
    unique_vertices = unique_combined[:, :3]
    unique_normals = unique_combined[:, 3:]
    indices = inverse_indices.reshape(-1, 3)
    return unique_vertices, unique_normals, indices

def split_faces(vertices: np.ndarray, normals: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return vertices[faces], normals[faces]


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

    vertices = np.array(vertices_list, dtype=np.float32)
    normals = np.array(normals_list, dtype=np.float32)
    faces = np.array(faces_list, dtype=np.uint32)
    if len(normals) != len(vertices):
        face_normals = compute_face_normals(vertices, faces)
        normals = compute_vertex_normals(face_normals, faces)
    vertices, normals = split_faces(vertices, normals, faces)
    return vertices, normals

def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix == '.stl':
        return load_stl(path)
    elif path.suffix == '.obj':
        return load_obj(path)
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")
