#pragma once

#include <c10/core/jabberwocky/Tensor.h>

#include <cstdlib>
#include <string>

namespace jabberwocky {

namespace {

  std::size_t compute_numel(const std::vector<int>& sizes) {
    std::size_t size = 1;
    for (auto i = decltype(sizes.size()){0}; i < sizes.size(); ++i) {
      size *= sizes[i];
    }
    return size;
  }

  bool check_shapes(const Tensor& lhs, const Tensor& rhs) {
    if (lhs.numel() != rhs.numel()) {
      return false;
    }

    return true;
  }

} // anonymous

Tensor full(const float f, const std::vector<int>& sizes) {
  const auto numel = compute_numel(sizes);
  return Tensor(numel, std::move(sizes));
}

Tensor add(const Tensor& lhs, const Tensor& rhs) {
  const auto has_valid_shapes = check_shapes(lhs, rhs);

  Tensor out{lhs.numel(), lhs.sizes()};

  for (auto i = decltype(lhs.numel()){0}; i < lhs.numel(); ++i) {
    out.storage()[i] = lhs.storage()[i] + rhs.storage()[i];
  }

  return out;
}

} // jabberwocky
