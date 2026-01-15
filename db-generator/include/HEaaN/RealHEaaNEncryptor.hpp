////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2024-2025 CryptoLab, Inc. All rights reserved.               //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "HEaaN/Ciphertext.hpp"
#include "HEaaN/Context.hpp"
#include "HEaaN/Plaintext.hpp"
#include "HEaaN/SecretKey.hpp"

// To separate slot-wise message for RealHEaaN from CoeffMessage
using RealMessage = std::vector<HEaaN::Real>;

namespace HEaaN {

class RealHEaaNEncryptorImpl;

class HEAAN_API RealHEaaNEncryptor {
public:
    RealHEaaNEncryptor(const Context context, const Context context_embed,
                       const Context context_real);

    // Encoding
    Plaintext encodeForRealHEaaN(const RealMessage &msg, u64 level,
                                 int r_counter = 0);
    Plaintext encodeForRealHEaaNWithScale(const RealMessage &msg, u64 level,
                                          Real scale);

    // Encryption
    void encryptToRealHEaaN(const RealMessage &msg,
                            const SecretKey &sk_real_embed,
                            Ciphertext &ctxt_real);
    void encryptToRealHEaaN(const RealMessage &msg,
                            const SecretKey &sk_real_embed,
                            Ciphertext &ctxt_real, u64 level,
                            int r_counter = 0);
    void encryptToRealHEaaN(const Plaintext &ptxt_real,
                            const SecretKey &sk_real_embed,
                            Ciphertext &ctxt_real);
    void encryptToRealHEaaNWithScale(const RealMessage &msg,
                                     const SecretKey &sk_real_embed,
                                     Ciphertext &ctxt_real, u64 level,
                                     Real scale);
    void encryptToRealHEaaN(const std::vector<RealMessage> &msg_vec,
                            const MSRLWESecretKey &sk_real_embed,
                            MSRLWECiphertext &ctxt_real);
    void encryptToRealHEaaN(const std::vector<RealMessage> &msg_vec,
                            const MSRLWESecretKey &sk_real_embed,
                            MSRLWECiphertext &ctxt_real, u64 level,
                            int r_counter = 0);
    void encryptToRealHEaaN(const std::vector<Plaintext> &ptxt_real_vec,
                            const MSRLWESecretKey &sk_real_embed,
                            MSRLWECiphertext &ctxt_real);
    void encryptToRealHEaaNWithScale(const std::vector<RealMessage> &msg_vec,
                                     const MSRLWESecretKey &sk_real_embed,
                                     MSRLWECiphertext &ctxt_real, u64 level,
                                     Real scale);

    // Context
    const Context &getContext();
    const Context &getContextEmbed();
    const Context &getContextReal();

private:
    std::shared_ptr<RealHEaaNEncryptorImpl> impl_;
};
} // namespace HEaaN
