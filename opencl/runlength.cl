/*
 * runlength.cl - column run length encoding of an image
 */

#define X (1)
#define Y (0)

__kernel void runlength(__global const uchar *image,
                        __local uchar *temp,
                        __global uchar *cur_runlength) {
                        /*int nhist,
                        __global volatile int *light_counts,
                        __global volatile int *dark_counts)*/
    // Load 8 bits from a byte of image into 8 *bytes* of temp
    int x = get_global_id(X);
    int y = get_global_id(Y);

    uchar8 mask = (uchar8)(1) << (uchar8)(7, 6, 5, 4, 3, 2, 1, 0);
    /*uchar8 pixels = (uchar8)(image[x + y * get_global_size(X)]) & mask;
    pixels = (uchar8)((pixels != (uchar8)(0)) & (uchar8)(0x1));
    vstore8(pixels, temp_x + temp_y * get_local_size(X), temp);*/
    int temp_x = get_local_id(X);
    int temp_y = get_local_id(Y);
    vstore8((uchar8)(1), temp_x + temp_y * get_local_size(X), temp);
    mem_fence(CLK_LOCAL_MEM_FENCE);

    // At each pixel, calculate the remaining length of the current run
    // (equivalent to distance to next run)
    // This is done by a prefix sum in reverse (suffix sum?) starting with
    // a 1 at every pixel and updating pixels above if they are part of the
    // same run as the run below
    uint k = 0;
    while ((1 << (k+1)) < get_local_size(Y)) {
        uint y0 =  2*temp_y      * (1 << k);
        uint y1 = (2*temp_y + 1) * (1 << k);
        if (y1 < get_local_size(Y)) {
            uchar pixels0 = image[x + (y1-1) * get_global_size(X)];
            uchar pixels1 = image[x + y1 * get_global_size(X)];
            //uchar8 is_run = (uchar8)((pixels0 == pixels1) & (uchar8)(0x1));
            uchar8 same_run = (uchar8)(~(pixels0 ^ pixels1));
            same_run >>= (uchar8)(7,6,5,4,3,2,1,0);
            same_run &= (uchar8)(0x1);
            uchar8 rl0 = vload8(temp_x + y0 * get_local_size(X), temp);
            uchar8 rl1 = vload8(temp_x + y1 * get_local_size(X), temp);
            rl1 *= same_run;
            rl1 *= (uchar8)(rl0 == (uchar8)(y1 - y0)) & (uchar8)(0x1);
            
            rl0 += rl1;
            vstore8(rl0, temp_x + y0 * get_local_size(X), temp);
            /*if (pixels[temp_x + (y1 - 1) * get_local_size(X)]
                == pixels[temp_x + y1 * get_local_size(X)])
                temp[temp_x + y0 * get_local_size(X)] +=
                    temp[temp_x + y1 * get_local_size(X)];*/
        }
        k++;
    }

    // Second step of suffix sum to propagate values to rest of Y positions
    /*k -= 2;
    while (k >= 0) {
        uint y0 = (2*temp_y + 1) * (1 << k);
        uint y1 = (2*temp_y + 2) * (1 << k);
        if (y1 < get_local_size(Y)) {
            if (pixels[temp_x + (y1 - 1) * get_local_size(X)]
                == pixels[temp_x + y1 * get_local_size(X)])
                temp[temp_x + y0 * get_local_size(X)] +=
                    temp[temp_x + y1 * get_local_size(X)];
        }
        k--;
    }*/


    mem_fence(CLK_LOCAL_MEM_FENCE);
    uchar8 run = vload8(temp_x + temp_y * get_local_size(X), temp);
    vstore8(run, x + y * get_global_size(X), cur_runlength);
}
