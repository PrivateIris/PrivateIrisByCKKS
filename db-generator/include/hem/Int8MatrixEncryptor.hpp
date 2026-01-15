////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2024-2025 CryptoLab, Inc. All rights reserved.               //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "HEaaN/HEaaN.hpp"
#include "HEaaN/RealHEaaNEncryptor.hpp"
#include "hem/CoeffEncryptor.hpp"
#include "hem/ExactBaseConverterU64U8.hpp"
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
// #define ERR_STD_DEV 3.2 // from CoeffEncryptor.hpp

using namespace HEaaN;

class Int8MatrixEncryptor {
public:
    Int8MatrixEncryptor(const size_t num_rows, const size_t num_cols,
                        Context &context_q, int level_q,
                        const std::vector<u64> &modulus_p);
    ~Int8MatrixEncryptor();

    void allocateBuffers(size_t buffer_num_rows);
    void deallocateBuffers();

    template <EncryptionType enc_type>
    void encryptInt8MatrixByBaseConv(std::vector<int8_t *> &d_A,
                                     std::vector<int8_t *> &d_B,
                                     SecretKeyBase<enc_type> &sk,
                                     const std::vector<CoeffPlaintext<int8_t>>
                                         &h_coeff_ptxt_vec, // row-major
                                     Device &device);

    template <EncryptionType enc_type>
    void encryptRealHEaaNInt8MatrixByBaseConv(
        std::vector<int8_t *> &d_A, std::vector<int8_t *> &d_B,
        SecretKeyBase<enc_type> &sk,
        const std::vector<CoeffPlaintext<int8_t>>
            &h_coeff_ptxt_vec, // row-major
        Device &device);

    void encryptRealHEaaNInt8MatrixBlockByBaseConv(
        std::vector<int8_t *> &h_A, std::vector<std::vector<int8_t *>> &h_B_vec,
        MSRLWESecretKey &sk_real_embed,
        const std::vector<CoeffPlaintext<int8_t>>
            &h_coeff_ptxt_vec, // row-major
        Device &device);

    template <EncryptionType enc_type>
    void convertInt8MatrixToCiphertextVector(
        std::vector<CiphertextBase<enc_type>> &ctxt_vec,
        std::vector<int8_t *> &mat_A, std::vector<int8_t *> &mat_B,
        const Device &device);

    void convertInt8MatrixBlocksToCiphertextVector(
        std::vector<MSRLWECiphertext> &ctxt_vec, std::vector<int8_t *> &mat_A,
        std::vector<std::vector<int8_t *>> &mat_B_vec, const Device &device);

    void setBaseConverter(ExactBaseConverterU64U8 *ebc) { ebc_ = ebc; }
    void setRealHEaaNEncryptor(RealHEaaNEncryptor *enc_real) {
        enc_real_ = enc_real;
    }
    inline std::vector<u64> getModulusP() {
        return std::vector<u64>(modulus_p_);
    }
    inline std::vector<u64> getModulusQ() {
        return std::vector<u64>(modulus_q_);
    }

private:
    void encryptZeroInt8MatrixByBaseConv(std::vector<int8_t *> &d_A,
                                         std::vector<int8_t *> &d_B,
                                         SecretKey &sk, Device &device);

    void encryptZeroInt8MatrixByBaseConv(std::vector<int8_t *> &d_A,
                                         std::vector<int8_t *> &d_B,
                                         MSRLWESecretKey &sk, Device &device);

    void encryptZeroRealHEaaNInt8MatrixByBaseConv(std::vector<int8_t *> &d_A,
                                                  std::vector<int8_t *> &d_B,
                                                  SecretKey &sk,
                                                  Device &device);

    void encryptZeroRealHEaaNInt8MatrixByBaseConv(std::vector<int8_t *> &d_A,
                                                  std::vector<int8_t *> &d_B,
                                                  MSRLWESecretKey &sk,
                                                  Device &device);

    void encryptZeroRealHEaaNInt8MatrixBlockByBaseConv(
        std::vector<int8_t *> &h_A, std::vector<std::vector<int8_t *>> &h_B,
        MSRLWESecretKey &sk, Device &device);

    template <EncryptionType enc_type>
    void convertCiphertextIntoInt64MatrixCUDA(CiphertextBase<enc_type> &ctxt,
                                              std::vector<i64 *> &mat_A,
                                              std::vector<i64 *> &mat_B,
                                              const size_t num_rows,
                                              const size_t num_cols,
                                              const size_t ctxt_idx);

    void moveInt8MatrixBufferCUDA(std::vector<int8_t *> d_mat_A_buffer,
                                  std::vector<int8_t *> d_mat_B_buffer,
                                  std::vector<int8_t *> d_mat_A,
                                  std::vector<int8_t *> d_mat_B,
                                  const size_t iter_idx,
                                  const size_t iter_num_rows);

    void addCoeffPlaintextIntoMatrixCUDA(
        const std::vector<CoeffPlaintext<int8_t>> &h_coeff_ptxt_vec,
        std::vector<int8_t *> &d_mat_B);

    void addCoeffPlaintextIntoMatrixBlockCUDA(
        const std::vector<CoeffPlaintext<int8_t>> &h_coeff_ptxt_vec,
        std::vector<std::vector<int8_t *>> &h_mat_B_block);

    void baseConversionFromU8ToU64(std::vector<i64 *> mat_q,
                                   std::vector<int8_t *> mat_p,
                                   const size_t mat_size);

    template <EncryptionType enc_type>
    void convertInt8MatrixToCiphertextVectorCPU(
        std::vector<CiphertextBase<enc_type>> &ctxt_vec,
        std::vector<int8_t *> &mat_A, std::vector<int8_t *> &mat_B,
        const Device &device);
    template <EncryptionType enc_type>
    void convertInt8MatrixToCiphertextVectorCUDA(
        std::vector<CiphertextBase<enc_type>> &ctxt_vec,
        std::vector<int8_t *> &mat_A, std::vector<int8_t *> &mat_B,
        const Device &device);

    void convertInt8MatrixBlocksToCiphertextVectorCPU(
        std::vector<MSRLWECiphertext> &ctxt_vec, std::vector<int8_t *> &mat_A,
        std::vector<std::vector<int8_t *>> &mat_B_vec, const Device &device);

    void loadInt8MatrixBufferCUDA(std::vector<int8_t *> d_mat_A_buffer,
                                  std::vector<int8_t *> d_mat_B_buffer,
                                  std::vector<int8_t *> d_mat_A,
                                  std::vector<int8_t *> d_mat_B,
                                  const size_t iter_idx,
                                  const size_t iter_num_rows);

    const size_t num_rows_;
    const size_t num_cols_;
    const size_t degree_;
    const size_t num_secret_;
    const size_t num_ct_;

    std::vector<u64> modulus_q_;
    size_t size_q_;
    std::vector<u64> modulus_p_;
    const size_t size_p_;

    u64 *h_modulus_q_;
    u64 *h_modulus_p_;

    ExactBaseConverterU64U8 *ebc_ = nullptr;
    RealHEaaNEncryptor *enc_real_ = nullptr;

    size_t buffer_num_rows_ = 0;

    std::vector<i64 *> h_A_q_;
    std::vector<i64 *> h_B_q_;

    std::vector<i64 *> d_A_q_;
    std::vector<i64 *> d_B_q_;

    std::vector<int8_t *> d_A_p_;
    std::vector<int8_t *> d_B_p_;

    u64 **d_pointer_arr_ = nullptr;

    std::vector<int8_t *> h_int8_row_maj_buffer_;
    int8_t *d_int8_row_maj_buffer_ = nullptr;

    // Added for block-wise encryption on MSRLWE
    std::vector<int8_t *> h_mat_block_buffer_vec_;
    int8_t *d_mat_block_buffer_ = nullptr;
    int8_t *d_mat_block_buffer2_ = nullptr;
};
