constant unsigned int A = 1664525;
constant unsigned int C = 1013904223;

KERNEL void kanungo_noise(GLOBAL_MEM UCHAR *img,
                          GLOBAL_MEM int *fg_dist,
                          GLOBAL_MEM int *bg_dist,
                          float nu,
                          float a0, float a,
                          float b0, float b,
                          uint seed) {
    int byte_x = get_global_id(0);
    int byte_y = get_global_id(1);
    int img_w = get_global_size(0);
    int bit_w = img_w * 8;
    UCHAR input = img[byte_x + img_w * byte_y];

    for (int bit = 0; bit < 8; bit++) {
        int img_x = byte_x * 8 + bit;
        UCHAR mask = 0x80U >> bit;
        float p;
        if (input & mask) {
            int dist = bg_dist[img_x + bit_w * byte_y];
            p = nu + a0 * exp(-a * dist * dist);
        }
        else {
            int dist = fg_dist[img_x + bit_w * byte_y];
            p = nu + b0 * exp(-b * dist * dist);
        }
        uint id = img_x + bit_w * byte_y;
        float rand = (float)((seed + id) * A + C) / (float)(0x100000000L);
        if (p > rand)
            input ^= mask;
    }

    img[byte_x + img_w * byte_y] = input;
}

KERNEL void patterns_3x3(GLOBAL_MEM UCHAR *img,
                         GLOBAL_MEM short *patterns) {
    int byte_x = get_global_id(0);
    int byte_y = get_global_id(1);
    int byte_w = get_global_size(0);
    int byte_h = get_global_size(1);

    for (int bit = 0; bit < 8; bit++) {
        int img_x = byte_x * 8 + bit;

        short pattern_num = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int x = img_x + dx;
                int y = byte_y + dy;
                if (0 <= x && x < byte_w && 0 <= y && y < byte_h) {
                    short pattern_bit = 1 << ((dx+1) + 3 * (dy+1));
                    int is_on = (img[x/8 + byte_w * y] >> (7 - (x%8))) & 0x1;
                    pattern_num += is_on * pattern_bit;
                }
            }
        }
        patterns[img_x + byte_w*8 * byte_y] = pattern_num;
    }
}
