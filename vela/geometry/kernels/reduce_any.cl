__kernel void reduce_any(
    __global const int* input,
    __global int* output,
    const unsigned int N)
{
    __local int ldata[256];

    unsigned int gid = get_global_id(0);
    unsigned int lid = get_local_id(0);
    unsigned int group_size = get_local_size(0);

    int value = (gid < N) ? input[gid] : 0;
    ldata[lid] = value;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int stride = group_size >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            ldata[lid] = ldata[lid] || ldata[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        output[get_group_id(0)] = ldata[0];
    }
}