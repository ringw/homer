typedef unsigned char UCHAR;

#ifdef CUDA
#define convert_int_rtn __float2int_rd
#define atomic_cmpxchg atomicCAS
#define atomic_inc(addr) atomicInc(addr, 0xFFFFFFFFU)
#define ATOMIC
#define barrier(x) __syncthreads()

struct int8 {
    int4 s0123;
    int4 s4567;
    inline __host__ __device__ int8 & operator>>=(const int8 &rhs) {
        this->s0123.x >>= rhs.s0123.x;
        this->s0123.y >>= rhs.s0123.y;
        this->s0123.z >>= rhs.s0123.z;
        this->s0123.w >>= rhs.s0123.w;
        this->s4567.x >>= rhs.s4567.x;
        this->s4567.y >>= rhs.s4567.y;
        this->s4567.z >>= rhs.s4567.z;
        this->s4567.w >>= rhs.s4567.w;
        return *this;
    }
    inline __host__ __device__ int8 & operator&=(const int8 &rhs) {
        this->s0123.x &= rhs.s0123.x;
        this->s0123.y &= rhs.s0123.y;
        this->s0123.z &= rhs.s0123.z;
        this->s0123.w &= rhs.s0123.w;
        this->s4567.x &= rhs.s4567.x;
        this->s4567.y &= rhs.s4567.y;
        this->s4567.z &= rhs.s4567.z;
        this->s4567.w &= rhs.s4567.w;
        return *this;
    }
    inline __host__ __device__ int8 & operator+=(const int8 &rhs) {
        this->s0123.x += rhs.s0123.x;
        this->s0123.y += rhs.s0123.y;
        this->s0123.z += rhs.s0123.z;
        this->s0123.w += rhs.s0123.w;
        this->s4567.x += rhs.s4567.x;
        this->s4567.y += rhs.s4567.y;
        this->s4567.z += rhs.s4567.z;
        this->s4567.w += rhs.s4567.w;
        return *this;
    }
    inline __host__ __device__ int8 operator^(const int8 rhs) {
        int4 a = s0123;
        int4 b = s4567;
        a.x ^= rhs.s0123.x;
        a.y ^= rhs.s0123.y;
        a.z ^= rhs.s0123.z;
        a.w ^= rhs.s0123.w;
        b.x ^= rhs.s4567.x;
        b.y ^= rhs.s4567.y;
        b.z ^= rhs.s4567.z;
        b.w ^= rhs.s4567.w;
        return (int8){a, b};
    }
    inline __host__ __device__ int8 operator~() {
        int4 a = make_int4(~s0123.x,~s0123.y,~s0123.z,~s0123.w);
        int4 b = make_int4(~s4567.x,~s4567.y,~s4567.z,~s4567.w);
        return (int8){a, b};
    }
};
inline __host__ __device__ float4 make_float4(int4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

#define fill_int8(x) \
    ((int8){make_int4(x,x,x,x), make_int4(x,x,x,x)})
#define make_int8(a,b,c,d,e,f,g,h) \
    ((int8){make_int4(a,b,c,d), make_int4(e,f,g,h)})
#define convert_float4 make_float4

inline __host__ __device__ float dot(float4 a, float4 b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
#endif

#ifndef CUDA
#define ATOMIC volatile
#define fill_int8(x) ((int8)(x))
#define make_int8(a,b,c,d,e,f,g,h) ((int8)(a,b,c,d,e,f,g,h))
#endif
