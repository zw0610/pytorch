#pragma once

#include <cstdlib>
#include <memory>

namespace jabberwocky {

struct Storage;

std::shared_ptr<Storage> get_storage(const std::size_t size) {
  return std::make_shared<Storage>(true, size);
}

struct Storage {

  Storage(const bool b, const std::size_t _size) : size_{_size} {
    storage_ = static_cast<float*>(std::malloc(size_*sizeof(int)));
  }
  ~Storage() {
    std::free(storage_);
  }

  std::size_t size() const { return size_; }
  float* const storage() { return storage_; }
  const float* const storage() const { return storage_; }

  const std::size_t size_;
  float* storage_ = nullptr;
};

} // jabberwocky