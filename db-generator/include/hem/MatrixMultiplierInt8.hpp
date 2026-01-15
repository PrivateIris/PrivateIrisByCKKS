////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2024-2025 CryptoLab, Inc. All rights reserved.               //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#pragma once
#include "HEaaN/HEaaN.hpp"
#include "HEaaN/device/Device.hpp"
#include <vector>

#define i8 std::int8_t

namespace HEaaN {
class MatrixMultiplierInt8 {
public:
    MatrixMultiplierInt8(const int m, const int k, const int n,
                         DeviceType device_type = DeviceType::GPU);
    ~MatrixMultiplierInt8();
    MatrixMultiplierInt8(const MatrixMultiplierInt8 &) = delete;
    MatrixMultiplierInt8 &operator=(const MatrixMultiplierInt8 &) = delete;
    MatrixMultiplierInt8(MatrixMultiplierInt8 &&) = delete;
    MatrixMultiplierInt8 &operator=(MatrixMultiplierInt8 &&) = delete;

    void singleRNSMultiplyMatrixAndMod(i8 *A, i8 *B, i8 *C, int m, int n, int k,
                                       u64 modulus, bool use_tc = false);
    void singleRNSMultiplyMatrixAndAddMod(i8 *A, i8 *B, i8 *C, int m, int n,
                                          int k, u64 modulus,
                                          bool use_tc = false);
    void addMatrixAndMod(const i8 *op, i8 *res, int m, int n, u64 modulus);
    void addMatrix(const i64 *op, i64 *res, int m, int n);
    void modMatrixInplace(i64 *op, int m, int n, u64 modulus);

    void subMatrixAndMultConstMod(i64 *A, i64 *B, int m, int n,
                                  const i64 mult_const, const u64 moduli);

private:
    void initializeCUDA();

    void singleRNSMultiplyMatrixAndModCUDA(i8 *A, i8 *B, i8 *C, int m, int n,
                                           int k, u64 modulus, bool use_tc);
    void singleRNSMultiplyMatrixAndAddModCUDA(i8 *A, i8 *B, i8 *C, int m, int n,
                                              int k, u64 modulus, bool use_tc);
    void addMatrixAndModCUDA(const i8 *op, i8 *res, int m, int n, u64 modulus);
    void addMatrixCUDA(const i64 *op, i64 *res, int m, int n);
    void modMatrixInplaceCUDA(i64 *op, int m, int n, u64 modulus);

    void subMatrixAndMultConstModCUDA(i64 *A, i64 *B, int m, int n,
                                      const i64 mult_const, const u64 moduli);

    int m_;
    int k_;
    int n_;

    i32 *c_int32_;

    DeviceType device_type_;
};
} // namespace HEaaN
