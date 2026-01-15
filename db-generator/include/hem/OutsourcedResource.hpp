////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2024-2025 CryptoLab, Inc. All rights reserved.               //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "HEaaN/HEaaN.hpp"

#include <optional>

namespace HEaaN {

class OutsourcedResource {
public:
    Context getContextMain() const;
    Context getRealContext() const;

    Context getCleanContext() const;
    Context getCleanRealContext() const;
    Context getCleanEmbedContext() const;

    const HomEvaluator &getCleanRealHomEvaluator() const;
    const HomEvaluator &getCleanEmbedHomEvaluator() const;

    RingSwitchKey getFromCleanRealSwitchKey() const;
    const Plaintext &getFromCleanRealMask() const;

    u64 getSwitchModulusTargetLevel() const;
    Real getSwitchModulusCompensation() const;
    Real getAdjustedScale() const;
    Real getPrimeScaleShift() const;

    void initializeResource(const SecretKey &sk,
                            const CudaDeviceIds &device_ids);

    void enableToBootstrap(HomEvaluator &eval, Bootstrapper &btp,
                           const CudaDeviceIds &device_ids);

    void loadEvaluationKeysAndBootConstants(const Device &device);

    void genFromRealMask(u64 level);

    void saveEvaluationKeysToFile(const std::string &path) const;
    void loadEvaluationKeysFromFile(const Context &context,
                                    const std::string &path);

private:
    static Context makeRealContext(const Context &context_base,
                                   const CudaDeviceIds &device_ids);

    static Context makeEmbedContext(const Context &context_base,
                                    const CudaDeviceIds &device_ids);

    static Context makeOutsourceRealContext(const Context &context_base,
                                            const CudaDeviceIds &device_ids);

    void genCleaningConstants();

    std::optional<Context> context_main_;

    std::optional<Context> context_real_;
    std::optional<Context> context_clean_;
    std::optional<Context> context_clean_real_;
    std::optional<Context> context_clean_embed_;

    std::optional<KeyPack> pack_obts_;
    std::optional<KeyPack> pack_obts_real_;
    std::optional<KeyPack> pack_small_;
    std::optional<KeyPack> pack_clean_real_;
    std::optional<KeyPack> pack_clean_embed_;

    std::optional<RingSwitchKey> to_real_swk_obts_;
    std::optional<RingSwitchKey> from_real_swk_obts_;

    std::optional<RingSwitchKey> from_clean_real_swk_;
    std::optional<Plaintext> from_clean_real_mask_;

    std::optional<Bootstrapper> btp_obts_;
    std::optional<HomEvaluator> eval_clean_real_;
    std::optional<HomEvaluator> eval_clean_embed_;

    u64 switch_modulus_from_level_ = 0;
    u64 switch_modulus_target_level_ = 0;
    Real switch_modulus_compensation_ = 1.0;
    Real adjusted_scale_ = 1.0;
    Real prime_scale_shift_ = 0.0;
};

} // namespace HEaaN
