# Adapted from astaka-pe's fantastic mesh_simplification repo (https://github.com/astaka-pe/mesh_simplification/tree/master/util/mesh.py)
import heapq
import numpy as np

OPTIM_VALENCE = 6
VALENCE_WEIGHT = 1

# Helper functions

def valence_weight(valence_new):
    valence_penalty = abs(valence_new - OPTIM_VALENCE) * VALENCE_WEIGHT + 1
    if valence_new == 3:
        valence_penalty *= 100000
    return valence_penalty

def compute_face_normals(vertices, faces):
    face_normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    norm = np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-24
    return face_normals / norm

def compute_face_center(vertices, faces):
    return np.sum(vertices[faces], 1) / 3.0

def compute_edges(faces):
    edges = set()
    for face in faces:
        for i in range(3):
            edge = (face[i], face[(i + 1) % 3])
            edge = tuple(sorted(list(edge)))
            edges.add(edge)
    return list(edges)

def compute_vertex_mappings(vertices, faces, edges):
    v2v: list[set[int]] = [set() for _ in range(len(vertices))]
    for i, (a, b) in enumerate(edges):
        v2v[a].add(b)
        v2v[b].add(a)

    v2f: list[set[int]] = [set() for _ in range(len(vertices))]
    for i, (a, b, c) in enumerate(faces):
        v2f[a].add(i)
        v2f[b].add(i)
        v2f[c].add(i)

    return v2v, v2f


# Mesh simplification

def simplify(vertices: np.ndarray, faces: np.ndarray, target_v: int) -> tuple[np.ndarray, np.ndarray]:
    vs, faces = vertices.copy(), faces.copy()
    face_normals = compute_face_normals(vs, faces)
    face_centers = compute_face_center(vs, faces)
    edges = compute_edges(faces)
    v2v, v2f = compute_vertex_mappings(vs, faces, edges)

    # 1. compute Q for each vertex
    Q: list[np.ndarray] = []
    for i, v in enumerate(vs):
        adjacent_faces = list(v2f[i])
        adjacent_face_centers = face_centers[adjacent_faces]
        adjacent_face_normals = face_normals[adjacent_faces]
        plane_distances = -np.sum(adjacent_face_normals * adjacent_face_centers, axis=1, keepdims=True)
        plane_abcds = np.concatenate([adjacent_face_normals, plane_distances], axis=1)
        Q.append(np.matmul(plane_abcds.T, plane_abcds))  # quadric error matrix is the outer product of the plane equations

    # 2. compute E for all edges and create heapq
    E_heap: list[tuple[float, int, int]] = []
    for i, (a, b) in enumerate(edges):
        Q_new = Q[a] + Q[b]
        Q_lp = np.eye(4)
        Q_lp[:3] = Q_new[:3]
        try:
            Q_lp_inv = np.linalg.inv(Q_lp)
            v4_new = np.matmul(Q_lp_inv, np.array([[0, 0, 0, 1]]).reshape(-1, 1)).reshape(-1)
        except np.linalg.LinAlgError:  # fall back to midpoint
            v4_new = np.r_[0.5 * (vs[a] + vs[b]), 1]

        valence_new = len((v2f[a] | v2f[b]) - (v2f[a] & v2f[b]))
        valence_penalty = valence_weight(valence_new)
        E_new = np.matmul(v4_new, np.matmul(Q_new, v4_new.T)) * valence_penalty
        heapq.heappush(E_heap, (E_new, a, b))

    # 3. collapse minimum-error vertex
    vertex_mask = np.ones(len(vs), dtype=bool)
    face_mask = np.ones(len(faces), dtype=bool)
    merges = [{i} for i in range(len(vs))]

    while np.sum(vertex_mask) > target_v and len(E_heap) > 0:
        _, a, b = heapq.heappop(E_heap)
        if vertex_mask[a] and vertex_mask[b]:  # skip if either vertex is already removed
            if len(v2v[a] & v2v[b]) == 2 and len(v2f[a] & v2f[b]) == 2:  # skip non-manifold and boundary edges
                collapse_edge(vs, a, b, vertex_mask, face_mask, merges, Q, E_heap, v2v, v2f)

    # 4. rebuild mesh
    vs = vs[vertex_mask]
    vertex_remaps = {j: i for i, vm in enumerate(merges) for j in vm}
    for i, f in enumerate(faces):
        for j in range(3):
            if f[j] in vertex_remaps:
                faces[i][j] = vertex_remaps[f[j]]

    faces = faces[face_mask]
    face_remaps = {i: j for i, j in enumerate(np.cumsum(vertex_mask) - 1)}
    for i, f in enumerate(faces):
        for j in range(3):
            faces[i][j] = face_remaps[f[j]]

    return vs, faces

def collapse_edge(vs, a, b, vertex_mask, face_mask, merges, Q, E_heap, v2v, v2f):
    vva, vvb = v2v[a] & v2v[b]  # we already checked that there would be two adjacent vertices in common
    merged_faces = v2f[a] & v2f[b]

    # Update vertex-to-face mapping
    v2f[a] = (v2f[a] | v2f[b]) - merged_faces
    v2f[b] = set()
    v2f[vva] = v2f[vva] - merged_faces
    v2f[vvb] = v2f[vvb] - merged_faces

    # Update vertex-to-vertex mapping
    v2v[a] = (v2v[a] | v2v[b]) - {a, b}
    for v in v2v[b]:
        if v != a:
            v2v[v] = (v2v[v] - {b}) | {a}
    v2v[b] = set()

    # Update merges
    merges[a] = (merges[a] | merges[b]) | {b}
    merges[b] = set()

    # Update masks and set a's new position
    vertex_mask[b] = False
    face_mask[list(merged_faces)] = False
    vs[a] = (vs[a] + vs[b]) / 2

    # recompute E
    for c in v2v[a]:
        v4_mid = np.r_[(vs[a] + vs[c]) / 2, 1]
        Q_new = Q[a] + Q[c]

        valence_new = len((v2f[a] | v2f[c]) - (v2f[a] & v2f[c]))
        valence_penalty = valence_weight(valence_new)
        E_new = np.matmul(v4_mid, np.matmul(Q_new, v4_mid.T)) * valence_penalty
        heapq.heappush(E_heap, (E_new, a, c))


if __name__ == '__main__':
    import sys
    from vela.helpers.urdf import load_obj
    assert len(sys.argv) == 3, "Usage: python simplify.py <input> <output>"

    vertices, _, faces = load_obj(sys.argv[1])
    vertices, faces = simplify(vertices, faces, 1000)

    with open(sys.argv[2], 'w') as fp:
        for x, y, z in vertices:
            fp.write(f'v {x:.8f} {y:.8f} {z:.8f}\n')
        for i0, i1, i2 in faces:
            fp.write(f'f {i0 + 1} {i1 + 1} {i2 + 1}\n')
