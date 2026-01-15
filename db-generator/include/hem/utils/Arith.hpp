////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2024-2025 CryptoLab, Inc. All rights reserved.               //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#pragma once
// TODO: Remove this file and use the original file from HEAAN library.
// This file is a copy of the original file from HEAAN library.
// The original file is located at HEAAN/src/Arith.cpp

#include "HEaaN/Integers.hpp"
#include "HEaaN/Real.hpp"

#include <algorithm>
#include <cmath>

#ifdef HEAAN_USE_ABSL_INT128
#include <absl/numeric/int128.h>
#endif

#ifdef __CUDACC__
#define CUDA_CALLABLE __device__
#define CUDA_CALLABLE_INLINE __inline__ __device__
#else
#define CUDA_CALLABLE
#define CUDA_CALLABLE_INLINE inline
#endif

namespace HEaaN {

#ifdef HEAAN_USE_ABSL_INT128
using u128 = absl::uint128;
using i128 = absl::int128;
#else
using u128 = unsigned __int128;
using i128 = __int128;
#endif

#define U64C(x) UINT64_C(x)
#define U128C(lo, hi) ((static_cast<u128>(U64C(hi)) << 64) + (lo))

inline u64 u128Lo(u128 x) { return static_cast<u64>(x); }

inline u64 u128Hi(u128 x) { return static_cast<u64>(x >> 64); }

inline u128 u128FromU64(u64 lo, u64 hi = U64ZERO) {
    return (static_cast<u128>(hi) << 64) | (static_cast<u128>(lo));
}

} // namespace HEaaN

namespace HEaaN::arith {

inline u64 countLeftZeroes(u64 op) {
#ifndef __has_builtin
#define __has_builtin(arg) 0
#endif
#if __has_builtin(__builtin_clzll)
    return static_cast<u64>(__builtin_clzll(op));
#elif _MSC_VER
    return static_cast<u64>(__lzcnt64(op));
#else
    // Algorithm: see "Hacker's delight" 2nd ed., section 5.13, algorithm 5-12.
    u64 n = 64;
    u64 tmp;
    tmp = op >> 32;
    if (tmp != 0) {
        n = n - 32;
        op = tmp;
    }
    tmp = op >> 16;
    if (tmp != 0) {
        n = n - 16;
        op = tmp;
    }
    tmp = op >> 8;
    if (tmp != 0) {
        n = n - 8;
        op = tmp;
    }
    tmp = op >> 4;
    if (tmp != 0) {
        n = n - 4;
        op = tmp;
    }
    tmp = op >> 2;
    if (tmp != 0) {
        n = n - 2;
        op = tmp;
    }
    tmp = op >> 1;
    if (tmp != 0)
        return n - 2;
    return n - op;
#endif
}

// Return bit width of the given parameter op.
// - If op equals to 0, then it returns 0. If op is positive, then it returns
// floor(log_2 op) + 1.
// - op is constraint to have unsigned long long type only.
// - TODO(TK): Replace this with std::bit_width (C++20).
inline u64 bitWidth(const u64 op) {
    return op ? U64C(64) - countLeftZeroes(op) : U64ZERO;
}

// Integral log2 with log2floor(0) := 0
inline u64 log2floor(const u64 op) { return op ? bitWidth(op) - 1 : U64ZERO; }

inline uint32_t bitReverse32(uint32_t x) {
    x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
    x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
    x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
    x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
    return ((x >> 16) | (x << 16));
}

inline uint32_t bitReverse(uint32_t x, u64 max_digits) {
    return bitReverse32(x) >> (32 - max_digits);
}

inline bool isPowerOfTwo(u64 op) { return op && (!(op & (op - 1))); }

template <typename T> void bitReverseArray(T *data, u64 n) {
    if (!(isPowerOfTwo(n)))
        return;

    for (u64 i = U64ONE, j = U64ZERO; i < n; ++i) {
        u64 bit = n >> 1;
        for (; j >= bit; bit >>= 1)
            j -= bit;

        j += bit;
        if (i < j)
            std::swap(data[i], data[j]);
    }
}

inline u64 subIfGE(u64 a, u64 b) { return (a >= b ? a - b : a); }

// Divide a 128 bit integer by a 64 bit integer and return the quotient.
// x_hi : The highest 64 bit of a 128 bit integer.
// x_lo : The lowest 64 bit of a 128 bit integer.
// returns Quotient of x divided by y.
inline u64 divide128By64Lo(u64 x_hi, u64 x_lo, u64 y) {
    return static_cast<u64>(u128FromU64(x_lo, x_hi) / y);
}

CUDA_CALLABLE_INLINE Real addZeroPointFive(Real x) {
    return x > 0 ? x + 0.5 : x - 0.5;
};

inline u128 mul64To128(const u64 op1, const u64 op2) {
    return static_cast<u128>(op1) * op2;
}

inline void mul64To128(u64 a, u64 b, u64 &hi, u64 &lo) {
    u128 mul = mul64To128(a, b);
    hi = u128Hi(mul);
    lo = u128Lo(mul);
}

inline u64 mul64To128Hi(const u64 op1, const u64 op2) {
    u128 mul = mul64To128(op1, op2);
    return u128Hi(mul);
}

inline u64 mulModSimple(u64 a, u64 b, u64 mod) {
    return static_cast<u64>(arith::mul64To128(a, b) % mod);
}

inline u64 powModSimple(u64 base, u64 expo, u64 mod) {
    u64 res = 1;
    while (expo > 0) {
        if (expo & 1) // if odd
            res = mulModSimple(res, base, mod);
        base = mulModSimple(base, base, mod);
        expo >>= 1;
    }

    return res;
}

inline u64 invModSimple(u64 a, u64 prime) {
    return powModSimple(a, prime - 2, prime);
}

inline u64 mulModLazy(const u64 x, const u64 y, const u64 y_barrett,
                      const u64 mod) {
    u64 q = arith::mul64To128Hi(x, y_barrett);
    return y * x - q * mod;
}

} // namespace HEaaN::arith
