import numpy as np
import pyopencl as cl
from pathlib import Path

class CollisionHandler:
    def __init__(self, vertex_arrays: list[np.ndarray], joints: list[tuple[int, int, str]]) -> None:
        kernel_dir = Path(__file__).parent / 'kernels'
        kernel_code = '\n\n'.join(cl_file.read_text() for cl_file in kernel_dir.glob('*.cl'))
        platforms = cl.get_platforms()
        self.ctx = cl.Context(dev_type=cl.device_type.ALL, properties=[(cl.context_properties.PLATFORM, platforms[0])])
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, kernel_code).build()
        self.meshes = vertex_arrays
        self.joints = joints

        # Create CL buffers and upload raw vertices
        self.vertex_buffers = []
        self.transformed_vertex_buffers = []

        for mesh in self.meshes:
            vertex_array = mesh.astype(np.float32).flatten()
            vertices_in = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=vertex_array)
            vertices_out = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, vertex_array.nbytes)
            self.vertex_buffers.append(vertices_in)
            self.transformed_vertex_buffers.append(vertices_out)

        # Create buffers for collision results
        self.collision_result_buffers = []
        self.collision_reduction_buffers = []
        self.collision_reduction_size = []

        for i, mesh in enumerate(self.meshes):
            T1_count = mesh.shape[0]
            T2_count = self.meshes[5].shape[0]  # Assuming we're always comparing with mesh 5
            max_groups = (T1_count * T2_count + 255) // 256
            results_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, T1_count * T2_count * np.int32().nbytes)
            self.collision_result_buffers.append(results_cl)

            reduction_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, max_groups * np.int32().nbytes)
            self.collision_reduction_buffers.append(reduction_cl)
            self.collision_reduction_size.append(max_groups)

    def __call__(self, transformations: list[np.ndarray]) -> np.ndarray:
        N = len(self.meshes)
        collision_matrix = np.zeros((N, N), dtype=bool)

        for i, transform in enumerate(transformations):
            matrix = np.array(transform, dtype=np.float32).T.flatten()
            matrix_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=matrix)
            vertex_count = len(self.meshes[i]) * 3
            a, b = self.vertex_buffers[i], self.transformed_vertex_buffers[i]
            self.prg.transform_vertices(self.queue, (vertex_count,), None, a, b, matrix_buf, np.int32(vertex_count))

        for i in range(N):
            T1_count = self.meshes[i].shape[0]
            T2_count = self.meshes[5].shape[0]
            global_size = T1_count * T2_count
            a, b = self.transformed_vertex_buffers[i], self.transformed_vertex_buffers[5]
            c, d = self.collision_result_buffers[i], self.collision_reduction_buffers[i]
            self.prg.triangle_triangle_intersection(self.queue, (global_size,), None, a, b, c, np.int32(T1_count), np.int32(T2_count))
            self.prg.reduce_any(self.queue, ((global_size + 255) // 256 * 256,), (256,), c, d, np.uint32(global_size))

        for i in range(N):
            collision_reduction_np = np.zeros(self.collision_reduction_size[i], dtype=np.int32)
            cl.enqueue_copy(self.queue, collision_reduction_np, self.collision_reduction_buffers[i])
            if np.any(collision_reduction_np):
                collision_matrix[i, 5] = True
                collision_matrix[5, i] = True

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
