////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2024-2025 CryptoLab, Inc. All rights reserved.               //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "HEaaN/HEaaN.hpp"
#include "hem/OutsourcedResource.hpp"

namespace HEaaN {

class EvaluationResource {
public:
    Context getContext() const;
    Context getRealContext() const;

    const HomEvaluator &getHomEvaluator() const;
    const Bootstrapper &getBootstrapper() const;
    const RingSwitchKey &getComposeKey() const;

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

    void genResource(const SecretKey &sk, const CudaDeviceIds &device_ids);

    void genComposeKey(const SecretKey &sk_hi, const SecretKey &sk_lo);

    void genFromRealMask(u64 level);

    void loadEvaluationKeysAndBootConstants(const Device &device);

    void saveEvaluationKeysToFile(const std::string &path) const;
    void loadEvaluationKeysFromFile(const Context &context,
                                    const std::string &path);

private:
    void isOutsourcedResourceGenerated() const;

    std::optional<Context> context_;
    std::optional<KeyPack> pack_;

    std::optional<HomEvaluator> eval_;
    std::optional<Bootstrapper> btp_;

    std::optional<RingSwitchKey> compose_key_;

    std::optional<OutsourcedResource> outsourced_resource_;
};

} // namespace HEaaN
