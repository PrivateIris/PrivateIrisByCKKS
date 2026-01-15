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

class BaseConverterU64U8Square {
public:
    BaseConverterU64U8Square(const std::vector<u64> modulus_q,
                             const std::vector<u64> modulus_p);
    ~BaseConverterU64U8Square();

    void initDevice();
    void allocateDeviceBuffer(const size_t size);
    void deallocateDeviceBuffer();

    void signedBaseConversionToU8Square(const std::vector<i64 *> &op,
                                        std::vector<i8 *> &res_upper,
                                        std::vector<i8 *> &res_lower,
                                        const Device device, const size_t size);

    void signedBaseConversionToU64(const std::vector<i8 *> &op_upper,
                                   const std::vector<i8 *> &op_lower,
                                   std::vector<i64 *> &res, const Device device,
                                   const size_t size);

    inline size_t getDeviceBufferSize() { return device_buffer_size_; }

private:
    void signedBaseConversionToU8SquareCPU(const std::vector<i64 *> &op,
                                           std::vector<i8 *> &res_upper,
                                           std::vector<i8 *> &res_lower,
                                           const size_t size);
    void signedBaseConversionToU8SquareCUDA(const std::vector<i64 *> &op,
                                            std::vector<i8 *> &res_upper,
                                            std::vector<i8 *> &res_lower,
                                            const size_t size);

    void signedBaseConversionToU64CPU(const std::vector<i8 *> &op_upper,
                                      const std::vector<i8 *> &op_lower,
                                      std::vector<i64 *> &res,
                                      const size_t size);
    void signedBaseConversionToU64CUDA(const std::vector<i8 *> &op_upper,
                                       const std::vector<i8 *> &op_lower,
                                       std::vector<i64 *> &res,
                                       const size_t size);

    // q: modulus chain for HEaaN
    // p: modulus chain for the modulus \prod_i p_i^2 where p_i <= 2^8
    const size_t size_q_;
    const size_t size_p_;
    u64 *host_q_;
    u64 *host_p_;
    u64 *host_p_sq_;
    double max_log_q_ = 0;

    bool is_device_init_ = false;
    u64 *device_q_;
    u64 *device_p_;
    u64 *device_p_sq_;

    // For conversion q -> p
    i64 *host_q_hat_inv_;
    i64 *host_prod_p_sq_mod_q_;
    i64 *host_prod_p_sq_mod_q_times_q_hat_inv_;
    i64 *host_negative_prod_q_inv_mod_p_sq_;
    std::vector<i64 *> host_q_inv_mod_p_sq_;
    bool lazy_reduction_q_to_p_ = false;

    i64 *device_q_hat_inv_ = nullptr;
    i64 *device_prod_p_sq_mod_q_ = nullptr;
    i64 *device_prod_p_sq_mod_q_times_q_hat_inv_ = nullptr;
    i64 *device_negative_prod_q_inv_mod_p_sq_ = nullptr;
    i64 **q_inv_mod_p_sq_device_ptr_ = nullptr;
    i64 **device_q_inv_mod_p_sq_ = nullptr;

    // For conversion p -> q
    i64 *host_p_sq_hat_inv_;
    i64 *host_prod_q_mod_p_sq_;
    i64 *host_prod_q_mod_p_sq_times_p_sq_hat_inv_;
    i64 *host_negative_prod_p_sq_inv_mod_q_;
    std::vector<i64 *> host_p_sq_inv_mod_q_;
    bool lazy_reduction_p_to_q_ = false;

    i64 *device_p_sq_hat_inv_ = nullptr;
    i64 *device_prod_q_mod_p_sq_ = nullptr;
    i64 *device_prod_q_mod_p_sq_times_p_sq_hat_inv_ = nullptr;
    i64 *device_negative_prod_p_sq_inv_mod_q_ = nullptr;
    i64 **p_sq_inv_mod_q_device_ptr_ = nullptr;
    i64 **device_p_sq_inv_mod_q_ = nullptr;

    // Device Buffer
    size_t device_buffer_size_ = 0;
    i64 *device_buffer_ = nullptr;

    i64 **val_q_device_ptr_;
    i64 **device_val_q_;
    i8 **val_p_upper_device_ptr_;
    i8 **device_val_p_upper_;
    i8 **val_p_lower_device_ptr_;
    i8 **device_val_p_lower_;
};
