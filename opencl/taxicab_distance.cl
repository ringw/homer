#define X (0)
#define Y (1)

#define MAXDIST (32)

__kernel void taxicab_distance_step(__global int *dists) {
    uint x = get_global_id(X);
    uint y = get_global_id(Y);
    uint w = get_global_size(X);
    uint h = get_global_size(Y);

    int8 dist = vload8(x + y * w, dists);
    
    int8 old_dist = dist;

    int left = MAXDIST;
    if (x > 0)
        left = dists[(x*8-1) + y * w * 8];
    int right = MAXDIST;
    if (x + 1 < w)
        right = dists[(x+1)*8 + y * w * 8];

    // Update our leftmost and rightmost pixel
    if (dist[0] > left + 1)
        dist[0] = left + 1;
    if (dist[7] > right + 1)
        dist[7] = right + 1;

    // Update each pixel with the values above and below
    int8 above = (int8)(MAXDIST);
    if (y > 0)
        above = vload8(x + (y-1) * w, dists);
    int8 below = (int8)(MAXDIST);
    if (y + 1 < h)
        below = vload8(x + (y+1) * w, dists);
    for (int b = 0; b < 8; b++) {
        if (dist[b] > above[b] + 1)
            dist[b] = above[b] + 1;
        if (dist[b] > below[b] + 1)
            dist[b] = below[b] + 1;
    }

    // Update adjacent pixels in this subrow by comparing to old_dist
    for (int b = 0; b < 7; b++)
        if (dist[b] > old_dist[b+1] + 1)
            dist[b] = old_dist[b+1] + 1;
    for (int b = 1; b < 8; b++)
        if (dist[b] > old_dist[b-1] + 1)
            dist[b] = old_dist[b-1] + 1;

    barrier(CLK_GLOBAL_MEM_FENCE);
    vstore8(dist, x + y * w, dists);
}
