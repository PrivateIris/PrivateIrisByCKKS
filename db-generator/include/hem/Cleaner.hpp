////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2024-2025 CryptoLab, Inc. All rights reserved.               //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "HEaaN/HEaaN.hpp"
#include "hem/EvaluationResource.hpp"

#include <vector>

namespace HEaaN {
class Cleaner {
public:
    Cleaner(const Context &context_hi, const Context &context_lo);

    void composeVector(const std::vector<Ciphertext> &ctxt_vec,
                       const RingSwitchKey &compose_key,
                       std::vector<Ciphertext> &ctxt_out_vec,
                       u64 batch_size) const;

    void halfBootAndCleanMultVector(const EvaluationResource &eval_resource,
                                    std::vector<Ciphertext> &ctxt_vec,
                                    double threshold) const;

    void halfBootAndCleanMultERF(const EvaluationResource &eval_resource,
                                 std::vector<Ciphertext> &ctxt_vec) const;

    void halfBootAndCleanMultERFBis(const EvaluationResource &eval_resource,
                                    std::vector<Ciphertext> &ctxt_vec) const;

    void halfBootAndCleanMultOutsource(const EvaluationResource &eval_resource,
                                       std::vector<Ciphertext> &ctxt_vec) const;

    void
    halfBootAndCleanMultOutsourceBis(const EvaluationResource &eval_resource,
                                     std::vector<Ciphertext> &ctxt_vec) const;

    void slotToCoeffMsg(const Message &msg_from, CoeffMessage &msg_to) const;

    void slotToCoeffMsgVector(const std::vector<Message> &msg_from,
                              std::vector<CoeffMessage> &msg_to) const;

    void coeffMsgToSlot(const CoeffMessage &msg_from, Message &msg_to) const;

    void coeffMsgToSlotVector(const std::vector<CoeffMessage> &msg_from,
                              std::vector<Message> &msg_to) const;

    void decomposeMsgVector(const std::vector<CoeffMessage> &msg_from,
                            std::vector<CoeffMessage> &msg_to,
                            u64 batch_size) const;

    void cleanSingle(const HomEvaluator &eval, Ciphertext &ctxt,
                     double threshold, u64 target_level) const;

    void cleanSingleERF(const HomEvaluator &eval, Ciphertext &ctxt,
                        u64 target_level) const;

    void cleanSingleERFBis(const HomEvaluator &eval, Ciphertext &ctxt,
                           double adjusted_compensation, Real prime_scale_shift,
                           u64 target_level) const;

    void cleanSingleOutsource(const HomEvaluator &eval, Ciphertext &ctxt,
                              double adjusted_compensation,
                              Real prime_scale_shift, u64 target_level) const;

    void cleanSingleOutsourceBis(const HomEvaluator &eval, Ciphertext &ctxt,
                                 double adjusted_compensation,
                                 Real prime_scale_shift,
                                 u64 target_level) const;

private:
    Context context_hi_;
    Context context_lo_;
};
} // namespace HEaaN
