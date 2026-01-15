////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2024-2025 CryptoLab, Inc. All rights reserved.               //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "HEaaN/Ciphertext.hpp"
#include "HEaaN/Context.hpp"
#include "HEaaN/Decryptor.hpp"
#include "HEaaN/Encryptor.hpp"
#include "HEaaN/KeyPack.hpp"
#include "HEaaN/Message.hpp"
#include "HEaaN/SecretKey.hpp"
#include "HEaaN/device/Device.hpp"

namespace HEaaN {

class HEDataBaseManager {
public:
    HEDataBaseManager(Context context);

    void deviceTo(std::vector<Ciphertext> &ctxt_database,
                  const Device &device) const;

    void convertColumnMajorMatrixToRowMajorRLWEMsgs(
        const double *database, u64 K, u64 N,
        std::vector<CoeffMessage> &msg_database) const;

    void encryptDataBase(std::vector<CoeffMessage> &msg_database,
                         const SecretKey &sk,
                         std::vector<Ciphertext> &ctxt_database, u64 level,
                         Real scale_factor, bool ntt_output) const;

    void encryptDataBase(std::vector<CoeffMessage> &msg_database,
                         const KeyPack &pack,
                         std::vector<Ciphertext> &ctxt_database, u64 level,
                         Real scale_factor, bool ntt_output) const;

    void encryptDataBase(const double *database, u64 K, u64 N,
                         const SecretKey &sk,
                         std::vector<Ciphertext> &ctxt_database, u64 level,
                         Real scale_factor, bool ntt_output) const;

    void encryptDataBase(const double *database, u64 K, u64 N,
                         const KeyPack &pack,
                         std::vector<Ciphertext> &ctxt_database, u64 level,
                         Real scale_factor, bool ntt_output) const;

    void decryptDataBase(const std::vector<Ciphertext> &ctxt_database,
                         const SecretKey &sk,
                         std::vector<CoeffMessage> &msg_database,
                         Real scale_factor, bool ntt_output) const;

    void convertRowMajorRLWEMsgsToColumnMajorMatrix(
        const std::vector<CoeffMessage> &msg_database, double *database, u64 K,
        u64 N) const;

    void decryptDataBase(const std::vector<Ciphertext> &ctxt_database,
                         const SecretKey &sk, double *database, u64 K, u64 N,
                         Real scale_factor, bool ntt_output) const;

    const Context &getContext() const { return context_; }

private:
    const Context context_;

    const Encryptor enc_;
    const Decryptor dec_;
};

} // namespace HEaaN
