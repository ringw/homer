__kernel void hough_line(__global const uchar *img,
                         __constant int *rho_res,
                         __constant float *tan_theta,
                         __global short8 *rho_inds) {
    int y = get_global_id(0);
    int xbin = get_global_id(1);
    int theta = get_global_id(2);

    float8 rho_vals = {0, 1, 2, 3, 4, 5, 6, 7};
    rho_vals += xbin*8;
    rho_vals *= -tan_theta[theta];
    rho_vals += y;
    rho_vals /= rho_res[0];

    short8 rho_ind = convert_short8(rho_vals);

    uchar8 img_bits = {1<<7, 1<<6, 1<<5, 1<<4, 1<<3, 1<<2, 1<<1, 1};
    img_bits &= img[y*get_global_size(1) + xbin];
    uchar8 shift = {7,6,5,4,3,2,1,0};
    img_bits >>= shift;
    
    short8 rho_mask = convert_short8(img_bits);
    rho_ind *= rho_mask;
    rho_ind &= ~(rho_ind >> 15); // Get rid of negative values

    float rho_max = ((float)get_global_size(0))**2 + ((float)get_global_size(1)*8)**2;
    rho_max = sqrt(rho_max);
    short8 rho_too_big = (short8)convert_short(rho_max);

    int out_ind = xbin + get_global_size(1)*(y + get_global_size(0)*theta);

    rho_inds[out_ind] = rho_ind;
}
