////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2024-2025 CryptoLab, Inc. All rights reserved.               //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#pragma once
#include "HEaaN/HEaaN.hpp"
#include "HEaaN/Integers.hpp"
#include "HEaaN/Real.hpp"
#include <vector>

using u128 = __uint128_t;
using i128 = __int128_t;
#define i8 std::int8_t

using namespace HEaaN;

class ExactBaseConverterU64U8 {
public:
    ExactBaseConverterU64U8(const std::vector<u64> modulus_q,
                            const std::vector<u64> modulus_p);
    ~ExactBaseConverterU64U8();

    void allocateBuffers(const size_t size);
    void allocateMemOptBuffers(const size_t size);
    void freeBuffers();
    void freeMemOptBuffers();

    void i128SignedExactBaseConversionToU8(std::vector<i64 *> &op,
                                           std::vector<i8 *> &res,
                                           const Device device,
                                           const size_t size);

    void i128SignedExactBaseConversionToU64(std::vector<i8 *> &op,
                                            std::vector<i64 *> &res,
                                            const Device device,
                                            const size_t size);

    void i128SignedCRTFromU64(std::vector<i64 *> &op, i128 *res,
                              const Device device, const size_t size);

    void i128SignedCRTFromU8(std::vector<i8 *> &op, i128 *res,
                             const Device device, const size_t size);

    void computeVFromU8(const std::vector<i8 *> &op, i64 *res,
                        const Device device, const size_t size,
                        const bool double_prec = false);

    void computeVFromU64(const std::vector<i64 *> &op, i64 *res,
                         const Device device, const size_t size,
                         const bool double_prec = false);

    void signedExactBaseConversionToU8(const std::vector<i64 *> &op,
                                       std::vector<i8 *> &res,
                                       const Device device, const size_t size,
                                       const bool fast_conv = false,
                                       const bool double_prec = false);

    void signedExactBaseConversionToU64(const std::vector<i8 *> &op,
                                        std::vector<i64 *> &res,
                                        const Device device, const size_t size,
                                        const bool fast_conv = false,
                                        const bool double_prec = false);

    void memOptSignedExactBaseConversionToU8(const std::vector<i64 *> &op,
                                             std::vector<i8 *> &res,
                                             const Device device,
                                             const size_t size,
                                             const bool fast_conv = false);

    void memOptSignedExactBaseConversionToU64(const std::vector<i8 *> &op,
                                              std::vector<i64 *> &res,
                                              const Device device,
                                              const size_t size,
                                              const bool fast_conv = false);

    void embedI8toI64Mod(i8 *op, i64 *res, u64 modulus, const Device device,
                         const size_t size);
    void embedI64toI8Mod(i64 *op, i8 *res, u64 modulus, const Device device,
                         const size_t size);

    void signedMod(i64 *&op, i64 *&res, const u64 modulus, const Device device,
                   const size_t size);

    inline bool getI128Overflow() { return i128_overflow_; }

    inline size_t getBufferSize() { return buffer_size_; }

    inline size_t getMemOptBufferSize() { return mem_opt_buffer_size_; }

private:
    void i128SignedExactBaseConversionToU8CUDA(std::vector<i64 *> &op,
                                               std::vector<i8 *> &res,
                                               const size_t size);

    void i128SignedExactBaseConversionToU8CPU(std::vector<i64 *> &op,
                                              std::vector<i8 *> &res,
                                              const size_t size);

    void i128SignedExactBaseConversionToU64CUDA(std::vector<i8 *> &op,
                                                std::vector<i64 *> &res,
                                                const size_t size);

    void i128SignedExactBaseConversionToU64CPU(std::vector<i8 *> &op,
                                               std::vector<i64 *> &res,
                                               const size_t size);

    void i128SignedCRTFromU64CPU(std::vector<i64 *> &op, i128 *res,
                                 const size_t size);

    void i128SignedCRTFromU8CPU(std::vector<i8 *> &op, i128 *res,
                                const size_t size);

    void i128SignedCRTFromU64CUDA(std::vector<i64 *> &op, i128 *res,
                                  const size_t size);

    void i128SignedCRTFromU8CUDA(std::vector<i8 *> &op, i128 *res,

                                 const size_t size);

    void computeVFromU8CUDA(const std::vector<i8 *> &op, i64 *res,
                            const size_t size, const bool double_prec = false);

    void computeVFromU8CPU(const std::vector<i8 *> &op, i64 *res,
                           const size_t size, const bool double_prec = false);

    void computeVFromU64CUDA(const std::vector<i64 *> &op, i64 *res,
                             const size_t size, const bool double_prec = false);

    void computeVFromU64CPU(const std::vector<i64 *> &op, i64 *res,
                            const size_t size, const bool double_prec = false);

    void signedExactBaseConversionToU8CUDA(const std::vector<i64 *> &op,
                                           std::vector<i8 *> &res,
                                           const size_t size,
                                           const bool fast_conv = false,
                                           const bool double_prec = false);

    void signedExactBaseConversionToU8CPU(const std::vector<i64 *> &op,
                                          std::vector<i8 *> &res,
                                          const size_t size,
                                          const bool fast_conv = false,
                                          const bool double_prec = false);

    void signedExactBaseConversionToU64CUDA(const std::vector<i8 *> &op,
                                            std::vector<i64 *> &res,
                                            const size_t size,
                                            const bool fast_conv = false,
                                            const bool double_prec = false);

    void signedExactBaseConversionToU64CPU(const std::vector<i8 *> &op,
                                           std::vector<i64 *> &res,
                                           const size_t size,
                                           const bool fast_conv = false,
                                           const bool double_prec = false);

    void memOptSignedExactBaseConversionToU8CUDA(const std::vector<i64 *> &op,
                                                 std::vector<i8 *> &res,
                                                 const size_t size,
                                                 const bool fast_conv = false);

    void memOptSignedExactBaseConversionToU64CUDA(const std::vector<i8 *> &op,
                                                  std::vector<i64 *> &res,
                                                  const size_t size,
                                                  const bool fast_conv = false);

    void embedI8toI64ModCUDA(i8 *op, i64 *res, const u64 modulus,
                             const size_t size);

    void embedI64toI8ModCUDA(i64 *op, i8 *res, const u64 modulus,
                             const size_t size);

    void signedModCUDA(i64 *&op, i64 *&res, const u64 modulus,
                       const size_t size);

    template <typename T>
    std::vector<T *> getBlockVec(const std::vector<T *> &mat,
                                 const size_t mat_size, const size_t iter_idx);

    // q: modulus chain for HEaaN
    // p: modulus chain of 8-bit words
    std::vector<u64> modulus_chain_q_;
    std::vector<u64> modulus_chain_p_;
    u64 *host_q_;
    u64 *host_p_;
    double max_log_q_ = 0;
    bool i128_overflow_ = false;

    u64 *device_q_;
    u64 *device_p_;

    // For conversion q -> p
    u128 host_prod_q_;
    u128 *host_q_hat_;
    u64 *host_q_hat_inv_;
    u64 *host_prod_p_mod_q_;
    u64 *host_prod_p_mod_q_times_q_hat_inv_;
    u64 *host_negative_prod_q_inv_mod_p_;
    std::vector<u64 *> host_q_inv_mod_p_;
    bool lazy_reduction_q_to_p_ = false;

    u128 *device_q_hat_;
    u64 *device_prod_p_mod_q_times_q_hat_inv_;
    u64 *device_negative_prod_q_inv_mod_p_;
    u64 **q_inv_mod_p_device_ptr_;
    u64 **device_q_inv_mod_p_;

    // For conversion p -> q
    u128 host_prod_p_;
    u128 *host_p_hat_;
    u64 *host_p_hat_inv_;
    u64 *host_prod_q_mod_p_;
    u64 *host_prod_q_mod_p_times_p_hat_inv_;
    u64 *host_negative_prod_p_inv_mod_q_;
    std::vector<u64 *> host_p_inv_mod_q_;
    bool lazy_reduction_p_to_q_ = false;

    u128 *device_p_hat_;
    u64 *device_prod_q_mod_p_times_p_hat_inv_;
    u64 *device_negative_prod_p_inv_mod_q_;
    u64 **p_inv_mod_q_device_ptr_;
    u64 **device_p_inv_mod_q_;

    // Device Buffer
    size_t buffer_size_;
    i64 **tmp_q_h_;
    i64 **tmp_q_d_;
    i8 **tmp_p_h_;
    i8 **tmp_p_d_;
    i64 *tmp64_;
    i128 *tmp128_;
    float *d_v_real_;
    i64 *d_v_;

    size_t mem_opt_buffer_size_;
    i64 *mem_opt_buffer_;
    i64 **mem_opt_ptr_q_h_;
    i64 **mem_opt_ptr_q_d_;
    i8 **mem_opt_ptr_p_h_;
    i8 **mem_opt_ptr_p_d_;
};
