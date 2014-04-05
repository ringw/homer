#define X (0)
#define Y (1)

#define MIN(a,b) (a < b ? a : b)

__kernel void taxicab_distance_step(__global int *dists) {
    uint x = get_global_id(X);
    uint y = get_global_id(Y);
    uint w = get_global_size(X);
    uint h = get_global_size(Y);

    int dist = dists[x + w * y];

    if (x > 0)
        dist = MIN(dist, dists[x-1 + w * y] + 1);
    if (x + 1 < w)
        dist = MIN(dist, dists[x+1 + w * y] + 1);
    if (y > 0)
        dist = MIN(dist, dists[x + w * (y-1)] + 1);
    if (y + 1 < h)
        dist = MIN(dist, dists[x + w * (y+1)] + 1);

    dists[x + w * y] = dist;
}
