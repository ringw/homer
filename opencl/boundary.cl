#define X (0)
#define L (1)
#define R (2)

__kernel void boundary_cost(__global const uint *dist,
                            int image_width,
                            int y0, int ystep, int numy,
                            int x0, int xstep, int numx,
                            __global float *costs) {
    int  x_ind = get_global_id(X);
    int yl_ind = get_global_id(L);
    int yr_ind = get_global_id(R);

    int xl = x0 + xstep * x_ind;
    int xr = x0 + xstep * (x_ind + 1);
    int yl = y0 + ystep * yl_ind;
    int yr = y0 + ystep * yr_ind;

    float sum_dt = 0.f;
    // Path weight term which priortizes avoiding dark pixels
    // by a large margin
    for (int x = xl; x < xr; x++) {
        int y = yl + (x - xl) * (yr - yl) / (xr - xl);
        uint dt = dist[x + y * image_width];
        sum_dt += 1.f / (1.f + dt);
    }
    // Priortize avoiding dark pixels
    // This coefficient needs to be fine tuned so the path can go through
    // small gaps between musical elements but usually avoids cutting through
    // actual parts of the music. Maybe there is a better edge cost formula
    // that doesn't need this fine tuning.
    sum_dt *= 25.f;
    // Actual edge Euclidean distance to prioritize horizontal paths
    float2 vec = (xr - xl, yl - yr);
    sum_dt += sqrt(dot(vec, vec));

    costs[x_ind * get_global_size(L) * get_global_size(R)
          + yl_ind * get_global_size(R)
          + yr_ind] = sum_dt;
}
