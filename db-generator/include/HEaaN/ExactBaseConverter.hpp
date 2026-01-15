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

// Exact base converter
// Assumption: \prod q_i < 2^128, \prod p_i < 2^128
namespace HEaaN {
class HEAAN_API ExactBaseConverter {
public:
    ExactBaseConverter(const std::vector<u64> modulus_chain_q,
                       const std::vector<u64> modulus_chain_p);
    ~ExactBaseConverter();

    void signedMod(i64 *&op, i64 *&res, const u64 modulus, const Device device,
                   const size_t size);

    void signedExactBaseConversion(std::vector<i64 *> &op,
                                   std::vector<i64 *> &res, const Device device,
                                   const size_t size);

    void exactBaseConversion(std::vector<u64 *> &op, std::vector<u64 *> &res,
                             const Device device, const size_t size);
    void signedCRT(const std::vector<i64 *> &op, __int128_t *&res,
                   const Device device, const size_t size);
    __uint128_t getProductQ() { return host_prod_q_; }
    __uint128_t *getHatQ() { return host_q_hat_; }
    u64 *getHatInverseQ() { return host_q_hat_inv_; }

private:
    void signedModCUDA(i64 *&op, i64 *&res, const u64 modulus,
                       const size_t size);
    void signedExactBaseConversionCUDA(std::vector<i64 *> &op,
                                       std::vector<i64 *> &res,
                                       const size_t size);

    void exactBaseConversionCUDA(std::vector<u64 *> &op,
                                 std::vector<u64 *> &res, const size_t size);
    void signedCRTCUDA(const std::vector<i64 *> &op, __int128_t *&res,
                       const size_t size);
    __uint128_t *host_q_hat_;
    __uint128_t host_prod_q_;
    std::vector<u64> modulus_chain_q_;
    std::vector<u64> modulus_chain_p_;
    u64 *host_q_hat_inv_;
    u64 *host_prod_p_mod_q_;
    u64 *host_prod_p_mod_q_times_q_hat_inv_;
    u64 *host_negative_prod_q_inv_mod_p_;
    u64 *host_p_;
    u64 *host_q_;

    u64 *device_modulus_chain_q_;
    u64 *device_modulus_chain_p_;
    u64 *device_prod_p_mod_q_times_q_hat_inv_;
    u64 *device_negative_prod_q_inv_mod_p_;
    __uint128_t *device_q_hat_;
    u64 *device_p_;
    u64 *device_q_;
};
} // namespace HEaaN
