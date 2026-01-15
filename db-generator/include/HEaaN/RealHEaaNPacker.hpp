////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2024-2025 CryptoLab, Inc. All rights reserved.               //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "HEaaN/Ciphertext.hpp"
#include "HEaaN/Context.hpp"
#include "HEaaN/SecretKey.hpp"
#include "HEaaN/impl/RealHEaaNPackerImpl.hpp"

namespace HEaaN {

class HomEvaluator;
class RingSwitchKey;

class HEAAN_API RealHEaaNPacker {
public:
    // coeff_emb = true implies that embedding for the ring packing is done in
    // non-NTT state, which requires backward/forward NTT before/after embedding
    // for the slot-wise encoding. To enable coeff_emb = false, the input
    // context should be based on a parameter with re_removei = true, and
    // constructed from makeNttEmbeddingContext function so that its NTT objects
    // are constructed with a proper root-of-unity.
    RealHEaaNPacker(const Context context, const Context context_embed,
                    const Context context_real, const Context context_hi,
                    const Context context_hi_embed,
                    const Context context_hi_real, const bool coeff_emb = true);

    // Allocate Device Buffer
    void allocateDeviceBuffer(const Device &device);

    // Swiching key generator
    RingSwitchKey genPackRealSwitchKey(SecretKey &sk_real,
                                       SecretKey &sk_hi_real);
    RingSwitchKey genPackEmbedSwitchKey(SecretKey &sk_real_embed,
                                        SecretKey &sk_hi_real_embed);

    // Packing
    Ciphertext packRealHEaaN(const std::vector<Ciphertext> &ctxt_vec,
                             RingSwitchKey &pack_swk);
    Ciphertext
    packRealHEaaNWithMaskMult(const std::vector<Ciphertext> &ctxt_vec,
                              std::vector<Message> &mask_vec,
                              RingSwitchKey &pack_swk);
    Ciphertext
    packRealHEaaNWithMaskMult(const std::vector<Ciphertext> &ctxt_vec,
                              std::vector<Message> &mask_vec, Real scale,
                              RingSwitchKey &pack_swk);
    Ciphertext
    packRealHEaaNWithConvToComplex(const std::vector<Ciphertext> &ctxt_vec,
                                   RingSwitchKey &pack_swk_embed);

private:
    std::shared_ptr<RealHEaaNPackerImpl> impl_;
};
} // namespace HEaaN
