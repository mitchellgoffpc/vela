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
