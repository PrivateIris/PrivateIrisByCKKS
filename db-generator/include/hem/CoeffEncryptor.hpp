////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2024-2025 CryptoLab, Inc. All rights reserved.               //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "HEaaN/HEaaN.hpp"
#include <memory.h>
#include <random>
#include <vector>

#define ERR_STD_DEV 3.2

using namespace HEaaN;

template <typename T> class CoeffPlaintext {
public:
    CoeffPlaintext(size_t degree, size_t level, std::vector<u64> modulus)
        : degree_(degree), modulus_(modulus) {
        for (size_t level_idx = 0; level_idx <= level; ++level_idx) {
            T *cur_poly = (T *)malloc(degree * sizeof(T));
            data_.push_back(cur_poly);
        }
    }

    void convertToHEaaN(Plaintext &out, bool outNTT = true);
    T *getPolyData(size_t level) { return data_[level]; }
    std::vector<u64> getModulus();
    size_t getLevel() { return data_.size() - 1; }
    size_t getDegree() { return degree_; }
    void multByInteger(i64 cnst_int);

private:
    std::vector<T *> data_;
    size_t degree_;
    std::vector<u64> modulus_;
};

template <typename T> class CoeffMSRLWECiphertext {
public:
    CoeffMSRLWECiphertext(size_t degree, size_t num_secret, size_t level,
                          std::vector<u64> modulus)
        : degree_(degree), num_secret_(num_secret), modulus_(modulus) {
        size_t num_polynomial = num_secret + 1;

        for (size_t poly_idx = 0; poly_idx < num_polynomial; ++poly_idx) {
            std::vector<T *> cur_rns_poly;
            for (size_t level_idx = 0; level_idx <= level; ++level_idx) {
                T *cur_poly = (T *)malloc(degree * sizeof(T));
                memset(cur_poly, 0, degree * sizeof(T));
                cur_rns_poly.push_back(cur_poly);
            }
            data_.push_back(cur_rns_poly);
        }
    }

    void convertToHEaaN(MSRLWECiphertext &out, bool outNTT = true);
    std::vector<u64> getModulus();
    size_t getNumSecret() { return num_secret_; }
    size_t getDegree() { return degree_; }
    size_t getLevel() { return data_[0].size() - 1; }
    std::vector<T *> getLevelledPolyData(size_t sk_idx) {
        return data_[sk_idx];
    }
    T *getPolyData(size_t sk_idx, size_t level) { return data_[sk_idx][level]; }

private:
    size_t degree_;
    size_t num_secret_;
    std::vector<u64> modulus_;
    std::vector<std::vector<T *>> data_;
};

template <typename T> class CoeffEncoder {
public:
    CoeffEncoder(std::vector<u64> modulus, size_t degree)
        : modulus_(modulus), degree_(degree) {}

    CoeffPlaintext<T> encodeWithScale(CoeffMessage msg, size_t level,
                                      double scale_factor);

private:
    std::vector<u64> modulus_;
    size_t degree_;
};

template <typename T> class CoeffMSRLWEEncryptor {
public:
    CoeffMSRLWEEncryptor(std::vector<u64> modulus) : modulus_(modulus) {
        std::random_device rd;
        gen_ = new std::mt19937(rd());
        dis_ = new std::uniform_int_distribution<u64>(0, u64(1) << 32);
    }

    void encrypt(std::vector<CoeffPlaintext<T>> &ptxt_vec,
                 const MSRLWESecretKey &sk, CoeffMSRLWECiphertext<T> &ctxt);

    void encrypt(CoeffPlaintext<T> &ptxt_vec, const SecretKey &sk,
                 CoeffMSRLWECiphertext<T> &ctxt);

private:
    u64 getUniformRandomU64();
    u64 getUniformRandomMod(u64 mod);
    void sampleUniformMod(std::vector<T *> &data, size_t size);
    void sampleDiscreteGaussian(i64 *data, size_t size);

    std::vector<u64> modulus_;
    std::mt19937 *gen_;
    std::uniform_int_distribution<u64> *dis_;
};

template class CoeffPlaintext<std::int8_t>;
template class CoeffEncoder<std::int8_t>;
template class CoeffMSRLWECiphertext<std::int8_t>;
template class CoeffMSRLWEEncryptor<std::int8_t>;

template class CoeffPlaintext<std::int16_t>;
template class CoeffEncoder<std::int16_t>;
template class CoeffMSRLWECiphertext<std::int16_t>;
template class CoeffMSRLWEEncryptor<std::int16_t>;

template class CoeffPlaintext<i32>;
template class CoeffEncoder<i32>;
template class CoeffMSRLWECiphertext<i32>;
template class CoeffMSRLWEEncryptor<i32>;

template class CoeffPlaintext<i64>;
template class CoeffEncoder<i64>;
template class CoeffMSRLWECiphertext<i64>;
template class CoeffMSRLWEEncryptor<i64>;
