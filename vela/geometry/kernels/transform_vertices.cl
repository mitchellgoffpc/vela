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
