////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2024-2025 CryptoLab, Inc. All rights reserved.               //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "HEaaN/HEaaN.hpp"
#include "HEaaN/RealHEaaNEncryptor.hpp"
#include "hem/BaseConverterU64U8Square.hpp"
#include "hem/CoeffEncryptor.hpp"
#include <iostream>

#define CHECK_CUDA(call)                                                       \
    {                                                                          \
        const cudaError_t error = call;                                        \
        if (error != cudaSuccess) {                                            \
            std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", ";     \
            std::cerr << "code: " << error                                     \
                      << ", reason: " << cudaGetErrorString(error) << "\n";    \
            exit(1);                                                           \
        }                                                                      \
    }

using namespace HEaaN;

class Int8SquareMatrixEncryptor {
public:
    Int8SquareMatrixEncryptor(const size_t num_rows, const size_t num_cols,
                              Context &context_q, int level_q,
                              const std::vector<u64> &modulus_p);
    ~Int8SquareMatrixEncryptor();

    void allocateDeviceBuffer(size_t buffer_num_rows);
    void deallocateDeviceBuffer();

    void encryptRealHEaaNInt8SquareMatrixBlockByBaseConv(
        std::vector<int8_t *> &h_A_upper, std::vector<int8_t *> &h_B_upper,
        std::vector<std::vector<int8_t *>> &h_B_upper_vec,
        std::vector<std::vector<int8_t *>> &h_B_lower_vec,
        MSRLWESecretKey &sk_real_embed,
        const std::vector<CoeffPlaintext<std::int16_t>> &h_coeff_ptxt_vec,
        Device &device);

    void encryptRealHEaaNInt8SquareMatrixBlockByBaseConvRescale(
        std::vector<int8_t *> &h_A_upper, std::vector<int8_t *> &h_B_upper,
        std::vector<std::vector<int8_t *>> &h_B_upper_vec,
        std::vector<std::vector<int8_t *>> &h_B_lower_vec,
        MSRLWESecretKey &sk_real_embed,
        const std::vector<CoeffPlaintext<std::int16_t>> &h_coeff_ptxt_vec,
        Device &device);

    void convertInt8SquareMatrixBlocksToCiphertextVector(
        std::vector<MSRLWECiphertext> &ctxt_vec,
        std::vector<int8_t *> &mat_A_upper, std::vector<int8_t *> &mat_A_lower,
        std::vector<std::vector<int8_t *>> &mat_B_upper_vec,
        std::vector<std::vector<int8_t *>> &mat_B_lower_vec,
        const Device &device);

    inline void setBaseConverter(BaseConverterU64U8Square *converter) {
        converter_ = converter;
    }
    inline void setRealHEaaNEncryptor(RealHEaaNEncryptor *enc_real) {
        enc_real_ = enc_real;
    }
    inline std::vector<u64> getModulusP() {
        return std::vector<u64>(modulus_p_);
    }
    inline std::vector<u64> getModulusQ() {
        return std::vector<u64>(modulus_q_);
    }

private:
    void encryptZeroRealHEaaNInt8SquareMatrixBlockByBaseConv(
        std::vector<int8_t *> &h_A_upper, std::vector<int8_t *> &h_A_lower,
        std::vector<std::vector<int8_t *>> &h_B_upper_vec,
        std::vector<std::vector<int8_t *>> &h_B_lower_vec,
        MSRLWESecretKey &sk_real_embed, Device &device);

    void encryptZeroRealHEaaNInt8SquareMatrixBlockByBaseConvRescale(
        std::vector<int8_t *> &h_A_upper, std::vector<int8_t *> &h_A_lower,
        std::vector<std::vector<int8_t *>> &h_B_upper_vec,
        std::vector<std::vector<int8_t *>> &h_B_lower_vec,
        MSRLWESecretKey &sk_real_embed, Device &device);

    void addCoeffPlaintextIntoMatrixBlockCUDA(
        const std::vector<CoeffPlaintext<int16_t>> &h_coeff_ptxt_vec,
        std::vector<std::vector<int8_t *>> &h_B_upper_block,
        std::vector<std::vector<int8_t *>> &h_B_lower_block);

    void convertCiphertextIntoMatrixInt64MatrixCUDA(MSRLWECiphertext &ctxt,
                                                    std::vector<i64 *> &mat_A,
                                                    std::vector<i64 *> &mat_B,
                                                    const size_t num_rows,
                                                    const size_t num_cols,
                                                    const size_t ctxt_idx);

    void convertInt8SquareMatrixBlocksToCiphertextVectorCPU(
        std::vector<MSRLWECiphertext> &ctxt_vec,
        std::vector<int8_t *> &mat_A_upper, std::vector<int8_t *> &mat_A_lower,
        std::vector<std::vector<int8_t *>> &mat_B_upper_vec,
        std::vector<std::vector<int8_t *>> &mat_B_lower_vec);

    const size_t num_rows_;
    const size_t num_cols_;
    const size_t degree_;
    const size_t num_secret_;
    const size_t num_ct_;

    std::vector<u64> modulus_q_;
    size_t size_q_;
    std::vector<u64> modulus_p_;
    size_t size_p_;

    u64 *h_modulus_q_;
    u64 *h_modulus_p_;
    u64 *h_modulus_p_sq_;

    BaseConverterU64U8Square *converter_ = nullptr;
    RealHEaaNEncryptor *enc_real_ = nullptr;

    std::vector<i64 *> h_A_q_;
    // std::vector<i64 *> h_B_q_;
    std::vector<std::vector<i64 *>> h_B_q_vec_;

    std::vector<i64 *> d_A_q_;
    std::vector<i64 *> d_B_q_;

    std::vector<int8_t *> d_A_p_upper_;
    std::vector<int8_t *> d_A_p_lower_;
    std::vector<int8_t *> d_B_p_upper_;
    std::vector<int8_t *> d_B_p_lower_;

    u64 **d_pointer_arr_ = nullptr;

    // Device Buffer
    size_t buffer_num_rows_ = 0;

    std::vector<int16_t *> h_mat_block_buffer_vec_;
    int16_t *d_mat_block_buffer_op_ = nullptr;
    int8_t *d_mat_block_buffer_res_upper_ = nullptr;
    int8_t *d_mat_block_buffer_res_lower_ = nullptr;
};
