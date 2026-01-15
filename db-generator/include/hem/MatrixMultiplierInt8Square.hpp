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
class MatrixMultiplierInt8Square {
public:
    MatrixMultiplierInt8Square(const int m, const int k, const int n,
                               DeviceType device_type = DeviceType::GPU);
    ~MatrixMultiplierInt8Square();

    void singleRNSMultiplyMatrixAndMod(i8 *A_upper, i8 *A_lower, i8 *B_upper,
                                       i8 *B_lower, i8 *C_upper, i8 *C_lower,
                                       int m, int n, int k, u64 modulus,
                                       bool use_tc = false);
    void singleRNSMultiplyMatrixAndAddMod(i8 *A_upper, i8 *A_lower, i8 *B_upper,
                                          i8 *B_lower, i8 *C_upper, i8 *C_lower,
                                          int m, int n, int k, u64 modulus,
                                          bool use_tc = false);
    void addMatrixAndModInt8Square(const i8 *op_upper, const i8 *op_lower,
                                   i8 *res_upper, i8 *res_lower, int m, int n,
                                   u64 modulus);

    void addMatrix(const i64 *op, i64 *res, int m, int n);
    void modMatrixInplace(i64 *op, int m, int n, u64 modulus);
    void subMatrixAndMultConstMod(i64 *A, i64 *B, int m, int n,
                                  const i64 mult_const, const u64 modulus);

private:
    void initializeCUDA();

    void singleRNSMultiplyMatrixAndModCUDA(i8 *A_upper, i8 *A_lower,
                                           i8 *B_upper, i8 *B_lower,
                                           i8 *C_upper, i8 *C_lower, int m,
                                           int n, int k, u64 modulus,
                                           bool use_tc = false);
    void singleRNSMultiplyMatrixAndAddModCUDA(i8 *A_upper, i8 *A_lower,
                                              i8 *B_upper, i8 *B_lower,
                                              i8 *C_upper, i8 *C_lower, int m,
                                              int n, int k, u64 modulus,
                                              bool use_tc = false);
    void addMatrixAndModInt8SquareCUDA(const i8 *op_upper, const i8 *op_lower,
                                       i8 *res_upper, i8 *res_lower, int m,
                                       int n, u64 modulus);

    void addMatrixCUDA(const i64 *op, i64 *res, int m, int n);
    void modMatrixInplaceCUDA(i64 *op, int m, int n, u64 modulus);
    void subMatrixAndMultConstModCUDA(i64 *op, i64 *res, int m, int n,
                                      const i64 mult_const, const u64 modulus);

    int m_;
    int k_;
    int n_;

    i32 *c_int32_;

    DeviceType device_type_;
};
} // namespace HEaaN
