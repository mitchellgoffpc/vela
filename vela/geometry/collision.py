import numpy as np
import pyopencl as cl

KERNEL_CODE = """
__kernel void transform_vertices(
    __global const float* vertices_in,
    __global float* vertices_out,
    __global const float* matrix,
    const int vertex_count)
{
    int gid = get_global_id(0);

    if (gid >= vertex_count)
        return;

    float x = vertices_in[gid * 3];
    float y = vertices_in[gid * 3 + 1];
    float z = vertices_in[gid * 3 + 2];
    float w = 1.0f;

    float tx = matrix[0] * x + matrix[4] * y + matrix[8]  * z + matrix[12] * w;
    float ty = matrix[1] * x + matrix[5] * y + matrix[9]  * z + matrix[13] * w;
    float tz = matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14] * w;
    float tw = matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15] * w;

    tx /= tw;
    ty /= tw;
    tz /= tw;

    vertices_out[gid * 3]     = tx;
    vertices_out[gid * 3 + 1] = ty;
    vertices_out[gid * 3 + 2] = tz;
}

float sign(float2 p1, float2 p2, float2 p3) {
    return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
}

int point_in_triangle(float2 p, float2 a, float2 b, float2 c) {
    float d1 = sign(p, a, b);
    float d2 = sign(p, b, c);
    float d3 = sign(p, c, a);
    int has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    int has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);
    return !(has_neg && has_pos);
}

int separating_axis_test(float3 axis, float3 T1[3], float3 T2[3]) {
    float min_T1 = dot(T1[0], axis);
    float max_T1 = min_T1;
    float min_T2 = dot(T2[0], axis);
    float max_T2 = min_T2;

    for (int i = 1; i < 3; i++) {
        float proj1 = dot(T1[i], axis);
        min_T1 = fmin(min_T1, proj1);
        max_T1 = fmax(max_T1, proj1);

        float proj2 = dot(T2[i], axis);
        min_T2 = fmin(min_T2, proj2);
        max_T2 = fmax(max_T2, proj2);
    }

    return (max_T1 >= min_T2) && (max_T2 >= min_T1);
}

int triangle_triangle_intersect(float3 T1[3], float3 T2[3]) {
    float3 edges_T1[3] = {
        T1[1] - T1[0],
        T1[2] - T1[1],
        T1[0] - T1[2]
    };
    float3 edges_T2[3] = {
        T2[1] - T2[0],
        T2[2] - T2[1],
        T2[0] - T2[2]
    };

    float3 N1 = cross(edges_T1[0], edges_T1[1]);
    float3 N2 = cross(edges_T2[0], edges_T2[1]);

    float N1_len = length(N1);
    float N2_len = length(N2);

    if (N1_len == 0.0f || N2_len == 0.0f) return 0;

    N1 /= N1_len;
    N2 /= N2_len;
    float d = dot(N1, T2[0] - T1[0]);

    if (fabs(d) < 1e-8f && (all(fabs(N1 - N2) < 1e-8f) || all(fabs(N1 + N2) < 1e-8f))) {
        int axis_idx;
        if (fabs(N1.z) > fabs(N1.x) && fabs(N1.z) > fabs(N1.y))
            axis_idx = 2;
        else if (fabs(N1.y) > fabs(N1.x))
            axis_idx = 1;
        else
            axis_idx = 0;

        float2 T1_2D[3], T2_2D[3];
        for (int i = 0; i < 3; i++) {
            if (axis_idx == 0) {
                T1_2D[i] = (float2)(T1[i].y, T1[i].z);
                T2_2D[i] = (float2)(T2[i].y, T2[i].z);
            } else if (axis_idx == 1) {
                T1_2D[i] = (float2)(T1[i].x, T1[i].z);
                T2_2D[i] = (float2)(T2[i].x, T2[i].z);
            } else {
                T1_2D[i] = (float2)(T1[i].x, T1[i].y);
                T2_2D[i] = (float2)(T2[i].x, T2[i].y);
            }
        }

        for (int i = 0; i < 3; i++) {
            if (point_in_triangle(T1_2D[i], T2_2D[0], T2_2D[1], T2_2D[2])) return 1;
            if (point_in_triangle(T2_2D[i], T1_2D[0], T1_2D[1], T1_2D[2])) return 1;
        }
        return 0;
    }

    float3 axes[11];
    int idx = 0;
    axes[idx++] = N1;
    axes[idx++] = N2;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float3 axis = cross(edges_T1[i], edges_T2[j]);
            if (length(axis) > 1e-8f) {
                axes[idx++] = normalize(axis);
            }
        }
    }

    for (int i = 0; i < idx; i++) {
        if (!separating_axis_test(axes[i], T1, T2)) return 0;
    }
    return 1;
}

__kernel void triangle_triangle_intersection(
    __global const float *T1_data,
    __global const float *T2_data,
    __global int *results,
    const int T1_count,
    const int T2_count)
{
    int gid = get_global_id(0);
    int T1_idx = gid / T2_count;
    int T2_idx = gid % T2_count;
    if (T1_idx >= T1_count) return;
    int i = T1_idx;
    int j = T2_idx;

    float3 T1[3], T2[3];

    for (int k = 0; k < 3; k++) {
        int idx_T1 = i * 9 + k * 3;
        int idx_T2 = j * 9 + k * 3;
        T1[k] = (float3)(T1_data[idx_T1], T1_data[idx_T1 + 1], T1_data[idx_T1 + 2]);
        T2[k] = (float3)(T2_data[idx_T2], T2_data[idx_T2 + 1], T2_data[idx_T2 + 2]);
    }

    results[i * T2_count + j] = triangle_triangle_intersect(T1, T2);
}
"""

class CollisionHandler:
    def __init__(self, vertex_arrays: list[np.ndarray]) -> None:
        platforms = cl.get_platforms()
        self.ctx = cl.Context(dev_type=cl.device_type.ALL,
                              properties=[(cl.context_properties.PLATFORM, platforms[0])])
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, KERNEL_CODE).build()
        self.meshes = vertex_arrays

    def __call__(self, transformations: list[np.ndarray]) -> np.ndarray:
        N = len(self.meshes)
        collision_matrix = np.zeros((N, N), dtype=bool)
        transformed_meshes = []

        for mesh, transform in zip(self.meshes, transformations):
            matrix = np.array(transform, dtype=np.float32).T.flatten()
            triangle_count = mesh.shape[0]
            vertex_count = triangle_count * 3
            vertex_array = mesh.astype(np.float32).reshape((-1, 3))
            vertex_array_flat = vertex_array.flatten()
            vertices_in = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=vertex_array_flat)
            vertices_out = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, vertex_array_flat.nbytes)
            matrix_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=matrix)
            self.prg.transform_vertices(self.queue, (vertex_count,), None,
                                        vertices_in, vertices_out, matrix_buf, np.int32(vertex_count))
            transformed_vertex_array_flat = np.empty_like(vertex_array_flat)
            cl.enqueue_copy(self.queue, transformed_vertex_array_flat, vertices_out)
            transformed_vertex_array = transformed_vertex_array_flat.reshape((-1, 9))
            transformed_meshes.append(transformed_vertex_array)

        for i in range(N):
            T1_array = transformed_meshes[i]
            T2_array = transformed_meshes[-1]
            T1_count = T1_array.shape[0]
            T2_count = T2_array.shape[0]
            if T1_count == 0 or T2_count == 0:
                continue
            T1_flat = T1_array.astype(np.float32).flatten()
            T2_flat = T2_array.astype(np.float32).flatten()
            T1_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=T1_flat)
            T2_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=T2_flat)
            results_np = np.zeros(T1_count * T2_count, dtype=np.int32)
            results_cl = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, results_np.nbytes)
            self.prg.triangle_triangle_intersection(self.queue, (T1_count * T2_count,), None,
                                                    T1_cl, T2_cl, results_cl,
                                                    np.int32(T1_count), np.int32(T2_count))
            cl.enqueue_copy(self.queue, results_np, results_cl)
            if np.any(results_np):
                collision_matrix[i, 0] = True
                collision_matrix[0, i] = True

                if i == 6:
                    t1_result = np.zeros_like(T1_flat)
                    t2_result = np.zeros_like(T2_flat)
                    cl.enqueue_copy(self.queue, t1_result, T1_cl)
                    cl.enqueue_copy(self.queue, t2_result, T2_cl)
                    t1_result = t1_result.reshape((-1, 3, 3))
                    t2_result = t2_result.reshape((-1, 3, 3))
                    results_np = results_np.reshape((T1_count, T2_count))
                    results_idx = np.where(results_np == 1)
                    for i, j in list(zip(results_idx[0], results_idx[1]))[:3]:
                        print(i, j)
                        print(t1_result[i])
                        print(t2_result[j])
                        print()
                    # t1_triangles = t1_result[np.any(results_np == 1, axis=1)]
                    # t2_triangles = t2_result[np.any(results_np == 1, axis=0)]
                    # print(t1_triangles.shape)
                    # print(t2_triangles.shape)
                    # print(t1_triangles[:3])
                    # print(t2_triangles[:3])

        # for i in range(N):
        #     for j in range(i + 1, N):
        #         T1_array = transformed_meshes[i]
        #         T2_array = transformed_meshes[j]
        #         T1_count = T1_array.shape[0]
        #         T2_count = T2_array.shape[0]
        #         if T1_count == 0 or T2_count == 0:
        #             continue
        #         T1_flat = T1_array.astype(np.float32).flatten()
        #         T2_flat = T2_array.astype(np.float32).flatten()
        #         T1_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=T1_flat)
        #         T2_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=T2_flat)
        #         results_np = np.zeros(T1_count * T2_count, dtype=np.int32)
        #         results_cl = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, results_np.nbytes)
        #         self.prg.triangle_triangle_intersection(self.queue, (T1_count * T2_count,), None,
        #                                                 T1_cl, T2_cl, results_cl,
        #                                                 np.int32(T1_count), np.int32(T2_count))
        #         cl.enqueue_copy(self.queue, results_np, results_cl)
        #         if np.any(results_np):
        #             collision_matrix[i, j] = True
        #             collision_matrix[j, i] = True
        return collision_matrix
