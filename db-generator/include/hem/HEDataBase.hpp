////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2024-2025 CryptoLab, Inc. All rights reserved.               //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "HEaaN/Ciphertext.hpp"

#include <ostream>
#include <vector>

namespace HEaaN {

class HEDataBase {
public:
    HEDataBase(const Context &context, size_t db_dim = 0, size_t db_size = 0);

    void to(Device &device);

    size_t getDim() const;
    size_t getSize() const;

    void setDim(size_t db_dim);
    void setSize(size_t db_size);

    const Context &getContext() const;

    Real getScaleFactor() const;
    void setScaleFactor(Real scale_factor);

    std::vector<Ciphertext> &getCtxtDatabase();

    const std::vector<Ciphertext> &getCtxtDatabase() const;

    void setCtxtDatabase(const std::vector<Ciphertext> &ctxt_database);

    void setCtxtDatabase(std::vector<Ciphertext> &&ctxt_database);

    void clear();

    void reserve(size_t size);

    void save(const std::string &path) const;

    void save(const std::ostream &stream) const;

    void load(const std::string &path);

    void load(const std::istream &stream);

private:
    size_t db_dim_;  // num_rows
    size_t db_size_; // num_cols

    Context context_;
    Real scale_factor_;
    std::vector<Ciphertext> ctxt_database_;
};

} // namespace HEaaN
