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

__kernel void triangle_triangle_intersection(
    __global const float* T1_array,
    __global const float* T2_array,
    __global int* results,
    const int T1_count,
    const int T2_count)
{
    int gid = get_global_id(0);
    int T1_idx = gid / T2_count;
    int T2_idx = gid % T2_count;

    if (T1_idx >= T1_count) return;

    __global const float* T1 = &T1_array[T1_idx * 9];
    __global const float* T2 = &T2_array[T2_idx * 9];

    float3 T1_0 = (float3)(T1[0], T1[1], T1[2]);
    float3 T1_1 = (float3)(T1[3], T1[4], T1[5]);
    float3 T1_2 = (float3)(T1[6], T1[7], T1[8]);

    float3 T2_0 = (float3)(T2[0], T2[1], T2[2]);
    float3 T2_1 = (float3)(T2[3], T2[4], T2[5]);
    float3 T2_2 = (float3)(T2[6], T2[7], T2[8]);

    float3 N1 = cross(T1_1 - T1_0, T1_2 - T1_0);
    float3 N2 = cross(T2_1 - T2_0, T2_2 - T2_0);
    float3 plane1_normal = normalize(N1);
    float3 plane2_normal = normalize(N2);

    if (length(N1) == 0.0f || length(N2) == 0.0f) {
        results[gid] = 0;
        return;
    }

    float dist_T2_0 = dot(T2_0 - T1_0, plane1_normal);
    float dist_T2_1 = dot(T2_1 - T1_0, plane1_normal);
    float dist_T2_2 = dot(T2_2 - T1_0, plane1_normal);

    if ((dist_T2_0 > 0.0f && dist_T2_1 > 0.0f && dist_T2_2 > 0.0f) ||
        (dist_T2_0 < 0.0f && dist_T2_1 < 0.0f && dist_T2_2 < 0.0f)) {
        results[gid] = 0;
        return;
    }

    float dist_T1_0 = dot(T1_0 - T2_0, plane2_normal);
    float dist_T1_1 = dot(T1_1 - T2_0, plane2_normal);
    float dist_T1_2 = dot(T1_2 - T2_0, plane2_normal);

    if ((dist_T1_0 > 0.0f && dist_T1_1 > 0.0f && dist_T1_2 > 0.0f) ||
        (dist_T1_0 < 0.0f && dist_T1_1 < 0.0f && dist_T1_2 < 0.0f)) {
        results[gid] = 0;
        return;
    }

    float3 D = cross(plane1_normal, plane2_normal);
    float len_D = length(D);

    if (len_D == 0.0f) {
        float abs_x = fabs(plane1_normal.x);
        float abs_y = fabs(plane1_normal.y);
        float abs_z = fabs(plane1_normal.z);
        int i0, i1;

        if (abs_x > abs_y && abs_x > abs_z) {
            i0 = 1; i1 = 2;
        } else if (abs_y > abs_z) {
            i0 = 0; i1 = 2;
        } else {
            i0 = 0; i1 = 1;
        }

        float2 T1_0_2D = (float2)(T1_0[i0], T1_0[i1]);
        float2 T1_1_2D = (float2)(T1_1[i0], T1_1[i1]);
        float2 T1_2_2D = (float2)(T1_2[i0], T1_2[i1]);

        float2 T2_0_2D = (float2)(T2_0[i0], T2_0[i1]);
        float2 T2_1_2D = (float2)(T2_1[i0], T2_1[i1]);
        float2 T2_2_2D = (float2)(T2_2[i0], T2_2[i1]);

        float2 axes[4] = {
            T1_1_2D - T1_0_2D,
            T1_2_2D - T1_1_2D,
            T2_1_2D - T2_0_2D,
            T2_2_2D - T2_1_2D
        };

        for (int i = 0; i < 4; i++) {
            float2 axis = (float2)(-axes[i].y, axes[i].x);
            float min1 = FLT_MAX, max1 = -FLT_MAX;
            float min2 = FLT_MAX, max2 = -FLT_MAX;

            float proj = dot(T1_0_2D, axis);
            min1 = fmin(min1, proj); max1 = fmax(max1, proj);
            proj = dot(T1_1_2D, axis);
            min1 = fmin(min1, proj); max1 = fmax(max1, proj);
            proj = dot(T1_2_2D, axis);
            min1 = fmin(min1, proj); max1 = fmax(max1, proj);

            proj = dot(T2_0_2D, axis);
            min2 = fmin(min2, proj); max2 = fmax(max2, proj);
            proj = dot(T2_1_2D, axis);
            min2 = fmin(min2, proj); max2 = fmax(max2, proj);
            proj = dot(T2_2_2D, axis);
            min2 = fmin(min2, proj); max2 = fmax(max2, proj);

            if (max1 < min2 || max2 < min1) {
                results[gid] = 0;
                return;
            }
        }

        results[gid] = 1;
        return;
    } else {
        float3 D_normalized = normalize(D);

        float proj_T1_0 = dot(T1_0, D_normalized);
        float proj_T1_1 = dot(T1_1, D_normalized);
        float proj_T1_2 = dot(T1_2, D_normalized);

        float min_T1 = fmin(fmin(proj_T1_0, proj_T1_1), proj_T1_2);
        float max_T1 = fmax(fmax(proj_T1_0, proj_T1_1), proj_T1_2);

        float proj_T2_0 = dot(T2_0, D_normalized);
        float proj_T2_1 = dot(T2_1, D_normalized);
        float proj_T2_2 = dot(T2_2, D_normalized);

        float min_T2 = fmin(fmin(proj_T2_0, proj_T2_1), proj_T2_2);
        float max_T2 = fmax(fmax(proj_T2_0, proj_T2_1), proj_T2_2);

        results[gid] = (max_T1 >= min_T2 && max_T2 >= min_T1) ? 1 : 0;
        return;
    }
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
            # T2_array = np.array([[0.0, 0.0, 0.0, 100.0, 100.0, 0.0, -100.0, 100.0, 0.0]], dtype=np.float32)
            # print(T1_array.shape, T2_array.shape)
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
