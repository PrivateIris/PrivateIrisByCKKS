////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2024-2025 CryptoLab, Inc. All rights reserved.               //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "HEaaN/Ciphertext.hpp"
#include "HEaaN/Context.hpp"
#include "HEaaN/RealHEaaNEncryptor.hpp"
#include "HEaaN/SecretKey.hpp"

namespace HEaaN {

class RealHEaaNDecryptorImpl;

class HEAAN_API RealHEaaNDecryptor {
public:
    RealHEaaNDecryptor(const Context context_embed);

    void decryptRealHEaaN(const Ciphertext &ctxt_real,
                          const SecretKey &sk_real_embed,
                          RealMessage &msg_real);
    void decryptRealHEaaN(const MSRLWECiphertext &ctxt_real,
                          const MSRLWESecretKey &sk_real_embed,
                          std::vector<RealMessage> &msg_real);
    void decryptRealHEaaNWithScale(const Ciphertext &ctxt_real,
                                   const SecretKey &sk_real_embed,
                                   RealMessage &msg_real, Real scale);
    void decryptRealHEaaNWithScale(const MSRLWECiphertext &ctxt_real,
                                   const MSRLWESecretKey &sk_real_embed,
                                   std::vector<RealMessage> &msg_real,
                                   Real scale);

private:
    std::shared_ptr<RealHEaaNDecryptorImpl> impl_;
};
} // namespace HEaaN
