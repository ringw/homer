// Linear congruential random number generator for platforms which only support
// 32-bit compare and swap
typedef unsigned int random_state;
constant unsigned int A = 1664525;
constant unsigned int C = 1013904223;
inline random_state rand_next(random_state rand) {
    return rand * A + C;
}
inline float rand_val(random_state rand) {
    return (float)rand / (float)0x100000000L;
}
inline random_state atom_rand(__global volatile random_state *rand) {
    random_state newVal, prevVal;
    do {
        prevVal = *rand;
        newVal = rand_next(prevVal);
    } while (atomic_cmpxchg(rand, prevVal, newVal) != prevVal);
    return newVal;
}
inline random_state atom_rand_l(__local volatile random_state *rand) {
    random_state newVal, prevVal;
    do {
        prevVal = *rand;
        newVal = rand_next(prevVal);
    } while (atomic_cmpxchg(rand, prevVal, newVal) != prevVal);
    return newVal;
}
// Split global rand with a second stream which can generate random numbers
// in parallel
inline random_state atom_split_rand(__global volatile random_state *rand) {
    random_state newVal, prevVal, retVal;
    do {
        prevVal = *rand;
        newVal = rand_next(prevVal + 0xDEADBEEF);
        retVal = rand_next(prevVal + 0x1);
    } while (atomic_cmpxchg(rand, prevVal, newVal) != prevVal);
    return retVal;
}
