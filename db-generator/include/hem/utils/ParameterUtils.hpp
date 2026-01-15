////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2024-2025 CryptoLab, Inc. All rights reserved.               //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "HEaaN/HEaaN.hpp"

namespace HEaaN {

inline ParameterPreset getCleaningParameter(const Context &context) {
    const auto parameter_name = getParameterName(context);
    if (parameter_name == "TONIC_3_BTS") {
        return ParameterPreset::TONIC_3_CLEANING;
    } else if (parameter_name == "WORLD_BTS") {
        return ParameterPreset::WORLD_CLEANING;
    } else {
        throw std::runtime_error("The parameter preset is not supported.");
    }
}

inline ParameterPreset getOutsourceRealParameter(const Context &context) {
    const auto parameter_name = getParameterName(context);
    if (parameter_name == "TONIC_3_BTS_OUTSOURCE") {
        return ParameterPreset::TONIC_3_BTS_OUTSOURCE_REAL;
    } else if (parameter_name == "WORLD_BTS_OUTSOURCE") {
        return ParameterPreset::WORLD_BTS_OUTSOURCE_REAL;
    } else {
        throw std::runtime_error("The parameter preset is not supported.");
    }
}

} // namespace HEaaN
