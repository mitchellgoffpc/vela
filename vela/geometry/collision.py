import numpy as np
import pyopencl as cl
from pathlib import Path

class CollisionHandler:
    def __init__(self, vertex_arrays: list[np.ndarray]) -> None:
        kernel_dir = Path(__file__).parent / 'kernels'
        kernel_code = '\n\n'.join(cl_file.read_text() for cl_file in kernel_dir.glob('*.cl'))
        platforms = cl.get_platforms()
        self.ctx = cl.Context(dev_type=cl.device_type.ALL, properties=[(cl.context_properties.PLATFORM, platforms[0])])
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, kernel_code).build()

        # Create CL buffers and upload raw vertices
        self.meshes = vertex_arrays
        self.vertex_buffers = []
        self.transformed_vertex_buffers = []

        for mesh in self.meshes:
            vertex_array = mesh.astype(np.float32).flatten()
            vertices_in = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=vertex_array)
            vertices_out = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, vertex_array.nbytes)
            self.vertex_buffers.append(vertices_in)
            self.transformed_vertex_buffers.append(vertices_out)

        # Create buffers for collision results
        self.collision_counts = np.full((len(self.meshes), len(self.meshes)), -1, dtype=int)
        self.collision_result_buffers = {}
        self.collision_reduction_buffers = {}
        self.collision_reduction_size = {}

        for i in range(len(self.meshes)):
            for j in range(i + 1, len(self.meshes)):
                T1_count = self.meshes[i].shape[0]
                T2_count = self.meshes[j].shape[0]
                max_groups = (T1_count * T2_count + 255) // 256
                results_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, T1_count * T2_count * np.int32().nbytes)
                self.collision_result_buffers[i, j] = results_cl

                reduction_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, max_groups * np.int32().nbytes)
                self.collision_reduction_buffers[i, j] = reduction_cl
                self.collision_reduction_size[i, j] = max_groups

    def __call__(self, transformations: list[np.ndarray]) -> np.ndarray:
        for i, transform in enumerate(transformations):
            matrix = np.array(transform, dtype=np.float32).T.flatten()
            matrix_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=matrix)
            vertex_count = len(self.meshes[i]) * 3
            a, b = self.vertex_buffers[i], self.transformed_vertex_buffers[i]
            self.prg.transform_vertices(self.queue, (vertex_count,), None, a, b, matrix_buf, np.int32(vertex_count))

        collision_matrix = np.zeros((len(self.meshes), len(self.meshes)), dtype=bool)
        for i in range(len(self.meshes)):
            for j in range(i + 1, len(self.meshes)):
                T1_count = len(self.meshes[i])
                T2_count = len(self.meshes[j])
                global_size = T1_count * T2_count
                a, b = self.transformed_vertex_buffers[i], self.transformed_vertex_buffers[j]
                c, d = self.collision_result_buffers[i, j], self.collision_reduction_buffers[i, j]
                self.prg.triangle_triangle_intersection(self.queue, (global_size,), None, a, b, c, np.int32(T1_count), np.int32(T2_count))
                self.prg.reduce_sum(self.queue, ((global_size + 255) // 256 * 256,), (256,), c, d, np.uint32(global_size))

                collision_reduction_np = np.zeros(self.collision_reduction_size[i, j], dtype=np.int32)
                cl.enqueue_copy(self.queue, collision_reduction_np, self.collision_reduction_buffers[i, j])
                collision_count = np.sum(collision_reduction_np)
                if self.collision_counts[i, j] == -1 or collision_count <= self.collision_counts[i, j]:
                    self.collision_counts[i, j] = collision_count
                else:  # collision count increased
                    collision_matrix[i, j] = True
                    collision_matrix[j, i] = True

        return collision_matrix
