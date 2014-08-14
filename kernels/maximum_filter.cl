__kernel void maximum_filter(const __global float *image,
                             __global float *maximum) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int w = get_global_size(0);
    int h = get_global_size(1);

    float val = image[x + w * y];
    if (x > 0) {
        if (y > 0 && val < image[(x-1) + w * (y-1)])
            val = 0;
        if (val < image[(x-1) + w * y])
            val = 0;
        if (y < h - 1 && val < image[(x-1) + w * (y+1)])
            val = 0;
    }
    if (y > 0 && val < image[x + w * (y-1)])
        val = 0;
    if (y < h - 1 && val < image[x + w * (y+1)])
        val = 0;
    if (x < w - 1) {
        if (y > 0 && val < image[(x+1) + w * (y-1)])
            val = 0;
        if (val < image[(x+1) + w * y])
            val = 0;
        if (y < h - 1 && val < image[(x+1) + w * (y+1)])
            val = 0;
    }

    maximum[x + y * w] = val;
}
