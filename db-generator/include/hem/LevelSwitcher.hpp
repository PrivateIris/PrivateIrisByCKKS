////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2021-2023 Crypto Lab Inc.                                    //
//                                                                            //
// - This file is part of HEaaN homomorphic encryption library.               //
// - HEaaN cannot be copied and/or distributed without the express permission //
//  of Crypto Lab Inc.                                                        //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#pragma once
#include "HEaaN/HEaaN.hpp"
#include "HEaaN/Integers.hpp"
#include "HEaaN/Real.hpp"
#include <vector>

using u128 = __uint128_t;
using i128 = __int128_t;

using namespace HEaaN;

class LevelSwitcher {
public:
    LevelSwitcher(const std::vector<u64> modulus_chain, int level_lo,
                  int level_hi);
    ~LevelSwitcher();

    void allocateDeviceBuffers(const size_t size);
    void deallocateDeviceBuffers();
    void allocateMemOptDeviceBuffers();
    void deallocateMemOptDeviceBuffers();
    inline size_t getBufferSize() { return buffer_size_; }

    void signedModUp(std::vector<i64 *> &op, const size_t size,
                     const Device device, const bool fast_conv = true);
    void signedModUp(i64 **op, const size_t size, const Device device,
                     const bool fast_conv = true);

    void signedModDown(std::vector<i64 *> &op, const size_t size,
                       const Device device, const bool fast_conv = true);
    void signedModDown(i64 **op, const size_t size, const Device device,
                       const bool fast_conv = true);

    void i128SignedCRTLo(std::vector<i64 *> &op, i128 *res, const size_t size,
                         const Device device);
    void i128SignedCRTLo(i64 **op, i128 *res, const size_t size,
                         const Device device);

    void i128SignedCRTHi(std::vector<i64 *> &op, i128 *res, const size_t size,
                         const Device device);
    void i128SignedCRTHi(i64 **op, i128 *res, const size_t size,
                         const Device device);

    void memOptSignedModUp(std::vector<i64 *> &op, const size_t size,
                           const Device device, const bool fast_conv = true);
    void memOptSignedModUp(i64 **op, const size_t size, const Device device,
                           const bool fast_conv = true);

    void memOptSignedModDown(std::vector<i64 *> &op, const size_t size,
                             const Device device, const bool fast_conv = true);
    void memOptSignedModDown(i64 **op, const size_t size, const Device device,
                             const bool fast_conv = true);

    bool getLazyReductionModUp() { return lazy_reduction_mod_up_; }
    bool getLazyReductionModDown() { return lazy_reduction_mod_down_; }
    void setLazyreductionModUp(bool lazy_reduction) {
        lazy_reduction_mod_up_ = lazy_reduction;
    }
    void setLazyreductionModDown(bool lazy_reduction) {
        lazy_reduction_mod_down_ = lazy_reduction;
    }

private:
    void signedModUpCPU(i64 **op, const size_t size,
                        const bool fast_conv = true);
    void signedModUpCUDA(i64 **op, const size_t size,
                         const bool fast_conv = true);

    void signedModDownCPU(i64 **op, const size_t size,
                          const bool fast_conv = true);
    void signedModDownCUDA(i64 **op, const size_t size,
                           const bool fast_conv = true);

    void i128SignedCRTLoCPU(i64 **op, i128 *res, const size_t size);
    void i128SignedCRTLoCUDA(i64 **op, i128 *res, const size_t size);

    void i128SignedCRTHiCPU(i64 **op, i128 *res, const size_t size);
    void i128SignedCRTHiCUDA(i64 **op, i128 *res, const size_t size);

    void memOptSignedModUpCUDA(i64 **op, const size_t size,
                               const bool fast_conv);
    void memOptSignedModDownCUDA(i64 **op, const size_t size,
                                 const bool fast_conv);

    void computeVForModUpCPU(i64 **op, i64 *v, const size_t size);
    void computeVForModUpCUDA(i64 **op, const size_t size);

    void computeVForModDownCPU(i64 **op, i64 *v, const size_t size);
    void computeVForModDownCUDA(i64 **op, const size_t size);

    std::vector<i64 *> getBlockVec(std::vector<i64 *> &op, const size_t op_size,
                                   const size_t iter_idx);

    std::vector<u64> modulus_chain_;
    int level_lo_ = 0;
    int level_hi_ = 0;
    bool i128_overflow_lo_ = false;
    bool i128_overflow_hi_ = false;
    bool lazy_reduction_mod_up_ = false;
    bool lazy_reduction_mod_down_ = false;

    u64 *host_q_;
    u64 *device_q_;

    size_t buffer_size_ = 0;
    i64 **device_buf_;
    i64 **host_device_buf_ptr_;

    i64 *device_v_;
    float *device_v_real_;

    i64 **host_op_ptr_;
    i64 **device_op_ptr_;

    // ModUp
    u64 *host_q_hat_lo_inv_mod_q_lo_;
    u64 **host_q_lo_inv_mod_q_hi_;
    u64 *host_prod_q_lo_mod_q_hi_;

    u64 *device_q_hat_lo_inv_mod_q_lo_;
    u64 **device_q_lo_inv_mod_q_hi_;
    u64 **host_q_lo_inv_mod_q_hi_device_ptr_;
    u64 *device_prod_q_lo_mod_q_hi_;

    // ModDown
    u64 *host_q_hat_hi_inv_mod_q_hi_;
    u64 *host_prod_q_hi_inv_mod_q_lo_;

    u64 *device_q_hat_hi_inv_mod_q_hi_;
    u64 **host_q_hi_inv_mod_q_lo_device_ptr_;
    u64 *device_prod_q_hi_inv_mod_q_lo_;

    // Memopt
    i64 **mem_opt_device_op_ptr_ = nullptr;

    u64 *host_q_hat_lo_mod_q_lo_;
    u64 **host_q_hi_inv_mod_q_lo_;

    u64 *device_q_hat_lo_mod_q_lo_;
    u64 **device_q_hi_inv_mod_q_lo_;
};
