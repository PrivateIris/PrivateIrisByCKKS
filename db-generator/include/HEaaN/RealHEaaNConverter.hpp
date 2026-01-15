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

namespace HEaaN {

class RealHEaaNConverterImpl;
class HomEvaluator;
class RingSwitchKey;

class HEAAN_API RealHEaaNConverter {
public:
    // coeff_emb = true implies that embedding to a ring of double degree (from
    // context to context_embed) is done in non-NTT state, which requires
    // backward/forward NTT before/after embedding for the slot-wise encoding.
    // To enable coeff_emb = false, the input context should be based on a
    // parameter with re_removei = true so that its NTT objects are constructed
    // with a proper root-of-unity.
    RealHEaaNConverter(const Context context, const Context context_embed,
                       const Context context_real, RingSwitchKey to_real_swk,
                       RingSwitchKey from_real_swk,
                       const bool coeff_emb = true);

    // Conversion
    void convertToRealHEaaN(const Ciphertext &ctxt, Ciphertext &ctxt_real);
    void convertFromRealHEaaN(const Ciphertext &ctxt_real, Ciphertext &ctxt);

    void embedRealHEaaN(const Ciphertext &ctxt, Ciphertext &ctxt_embed,
                        const bool coeff_emb = true);
    void convertToHalfDegree(const Ciphertext &ctxt_embed, Ciphertext &ctxt,
                             const bool coeff_emb = true);

    // Context
    const Context &getContext();
    const Context &getContextEmbed();
    const Context &getContextReal();

private:
    std::shared_ptr<RealHEaaNConverterImpl> impl_;
};
} // namespace HEaaN
