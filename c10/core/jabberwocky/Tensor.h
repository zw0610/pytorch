#pragma once

#include <cstdlib>
#include <memory>

#include <c10/core/jabberwocky/View.h>
#include <c10/core/jabberwocky/Storage.h>

namespace jabberwocky {

struct Tensor {

  Tensor(float* const storage() { return storage_; }
  const float* const storage() const { return storage_; }mo
    const std::size_t numel
  , const std::vector<int>& sizes)
  : view_{numel, sizes} {
    storage_ = get_storage(numel);
  }

  std::size_t numel() const { return view_.numel(); }
  const std::vector<int>& sizes() const { return view_.sizes(); }
  const std::vector<int>& strides() const { return view_.strides(); }
  float* const storage() { return storage_->storage(); }
  const float* const storage() const { return storage_->storage(); }

  const View view_;
  std::shared_ptr<Storage> storage_ = nullptr;
};

} // jabberwocky