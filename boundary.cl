#define X (0)
#define L (1)
#define R (2)

__kernel void boundary_cost(__global const float *dist,
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

    // Sum distance transform along path of line, and add 1
    float sum_dt = 1.0f;
    for (int x = xl; x < xr; x++) {
        int y = yl + (x - xl) * (yr - yl) / (xr - xl);
        sum_dt += dist[x + y * image_width];
    }

    costs[x_ind * get_global_size(L) * get_global_size(R)
          + yl_ind * get_global_size(R)
          + yr_ind] = 1.0f / sum_dt;
}
