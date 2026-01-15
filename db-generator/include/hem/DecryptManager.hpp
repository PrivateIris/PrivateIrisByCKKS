////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2024-2025 CryptoLab, Inc. All rights reserved.               //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "HEaaN/HEaaN.hpp"

#include <vector>

namespace HEaaN {

SecretKey genSecretKeyForDecrypt(const SecretKey &sk);

class DecryptManager {
public:
    DecryptManager(const SecretKey &sk);

    void decrypt(const std::vector<Ciphertext> &ctxt_vec,
                 std::vector<Message> &msg_vec) const;

private:
    SecretKey sk_;
    Decryptor dec_;
};

} // namespace HEaaN
