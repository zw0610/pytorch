#pragma once

#include <cstdlib>
#include <vector>

namespace jabberwocky {

struct View {

  View(
    const std::size_t _numel
  , const std::vector<int>& _sizes)
  : sizes_{_sizes}
  , numel_{_numel} {
  }

  View(
    const std::size_t _numel
  , const std::vector<int>& _sizes
  , const std::vector<int>& _strides)
  : numel_{_numel}
  , sizes_{_sizes}
  , strides_{_strides} { }

  std::size_t numel() const { return numel_; }
  const std::vector<int>& sizes() const { return sizes_; }
  const std::vector<int>& strides() const { return strides_; }

private:
  std::vector<int> sizes_;
  std::vector<int> strides_;
  std::size_t numel_;
};

} // jabberwocky