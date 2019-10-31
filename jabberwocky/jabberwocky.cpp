#include <cstdint>
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <string>
#include <memory>
#include <vector>
#include <atomic>
#include <utility>
#include <iostream>
#include <exception>

namespace jabberwocky {

/*
TODO: setStorage, setView, tensor copy/move semantics, borrow semantics
refcount views? o.w. copy is expensive, but shouldn't have to copy often
o.w. putting view in refcount makes the refcounter "own" that view
o.w. could put view back in tensor (that's probably best)
-> yeah put view in tensor, if refcounted and storage is changed then
-> adjust refcounter you're borrowed from
update interface functions to use refcounted and then borrow before going internal
let views be created and test they work properly and validate their perf
refactor out add kernel and apples to apples compare (so just reviewing glue)
add sizes and strides to tensor and kernel, compare with optimal collapsed
provide a mechanism for "graph mode" to capture and run optimally
show how that works and show the cost
*/

// struct Storage {

//   Storage(const std::size_t _size) : size_{_size} {
//     storage_ = static_cast<float*>(std::malloc(size_));
//   }
//   ~Storage() {
//     std::free(storage_);
//   }

//   // Atomic refcounting
//   int acquire() {
//     return refs_.fetch_add(1, std::memory_order_relaxed);
//   }
//   bool release() {
//     const auto prev = refs_.fetch_sub(1, std::memory_order_acq_rel);
//     if (prev == 1) {
//       return true;
//     }

//     return false;
//   }

//   std::size_t size() const { return size_; }
//   float* const storage() { return storage_; }
//   const float* const storage() const { return storage_; }

//   const std::size_t size_;
//   float* storage_ = nullptr;
//   std::atomic<uint64_t> refs_{1};
// };

struct Storage {
  Storage(const std::size_t _size) : size_{_size} {
    storage_ = (float*)std::malloc(size_);
  }
  ~Storage() {
    std::free(storage_);
  }

  const std::size_t size_;
  float* storage_ = nullptr;
};

struct Tensor {

  Tensor(
    const std::size_t _numel)
  : numel_{_numel} {
    const auto size = sizeof(float) * numel_;
    // storage_ = new Storage{size};
    storage_ = std::make_shared<Storage>(size);
  }

  // Tensor(
  //   const std::size_t _numel
  // , std::shared_ptr<Storage> _storage)
  // : numel_{_numel}
  // , storage_{_storage} { }

  ~Tensor() {
    release();
  }

  // Copy and copy assignment operator
  Tensor(const Tensor& other) {
    copy(other);
  }
  Tensor& operator=(const Tensor& other) {
    copy(other);
    return *this;
  }
  void copy(const Tensor& other) {
    release();
    numel_ = other.numel_;
    storage_ = other.storage_;
    acquire();
  }

  // Move and move assignment operator
  Tensor(Tensor&& other) {
    swap(std::move(other));
  }
  Tensor& operator=(Tensor&& other) {
    swap(std::move(other));
    return *this;
  }
  void swap(Tensor&& other) {
    std::swap(numel_, other.numel_);
    std::swap(storage_, other.storage_);
  }

  // Atomic refcounting (for storage)
  void acquire() {
    // const auto prev = storage_->acquire();
    // TODO: validate prev >= 1
  }
  void release() {
    // if (storage_ && storage_->release()) {
    //   std:free(storage_);
    // }
  }

  std::size_t numel() const { return numel_; }
  float* const storage() { return storage_->storage_; }
  const float* const storage() const { return storage_->storage_; }

  std::size_t numel_ = 0;
  std::shared_ptr<Storage> storage_ = nullptr;
  // Storage* storage_ = nullptr;
};

struct BorrowedTensor {

  BorrowedTensor(Tensor& t) : t_{&t} { }

  std::size_t numel() const { return t_->numel_; }
  float* const storage() { return t_->storage(); }
  const float* const storage() const { return t_->storage(); }

  Tensor* t_ = nullptr;
};

bool check_shapes(const Tensor& lhs, const Tensor& rhs) {
  if (lhs.numel() != rhs.numel()) {
    return false;
  }

  return true;
}

Tensor full(const float f, const std::size_t numel) {
  Tensor t{numel};

  for (auto i = decltype(t.numel()){0}; i < t.numel(); ++i) {
    t.storage()[i] = f;
  }

  return t;
}

void add_unchecked(Tensor& out, const Tensor& lhs, const Tensor& rhs) {
  auto* const out_storage = out.storage();
  const auto* const lhs_storage = lhs.storage();
  const auto* const rhs_storage = rhs.storage();

  for (auto i = decltype(lhs.numel()){0}; i < lhs.numel(); ++i) {
    out_storage[i] = lhs_storage[i] + rhs_storage[i];
  }
}

Tensor add(const Tensor& lhs, const Tensor& rhs) {
  const auto has_valid_shapes = check_shapes(lhs, rhs);

  Tensor out{lhs.numel()};
  add_unchecked(out, lhs, rhs);
  return out;
}

void print(const Tensor& t) {
  std::cout << "Tensor : ";

  for (auto i = decltype(t.numel()){0}; i < t.numel(); ++i) {
    std::cout << t.storage()[i] << " " << std::endl;
  }
}

} // jabberwocky

typedef std::chrono::nanoseconds ns;
constexpr uint64_t warmup_iters = 1000;
constexpr uint64_t iters = 20000000;
constexpr int max_value = 50;

int rand(const int max_value) {
  return 1 + std::rand()/((RAND_MAX + 1u)/max_value);
}

int main() {
  std::srand(std::time(0));
  const auto size = ::rand(max_value);

  jabberwocky::Tensor a = jabberwocky::full(1.f, size);
  jabberwocky::Tensor b = jabberwocky::full(2.f, size);
  // jabberwocky::Tensor c = jabberwocky::full(2.f, size);

  // Warm-up loop
  for (auto i = decltype(warmup_iters){0}; i < warmup_iters; ++i) {
    auto c = jabberwocky::add(a, b);
  }

  const auto start = std::chrono::high_resolution_clock::now();

  for (auto i = decltype(iters){0}; i < iters; ++i) {
    auto c = jabberwocky::add(a, b);
  }

  const auto end = std::chrono::high_resolution_clock::now();
  const auto elapsed = end - start;
  const auto elapsed_ns = std::chrono::duration_cast<ns>(elapsed);
  const std::size_t flops = size * iters;
  const std::size_t bytes = sizeof(float) * 3 * size * iters;

  std::cout << "Ran in " << elapsed_ns.count() << "ns\n" << std::endl;
  std::cout << "Array size: " << size << "\n" << std::endl;
  std::cout << "Each iteration took ~" << elapsed_ns.count()/iters << "ns\n" << std::endl;
  std::cout << "(perceived) Flops per nanonsecond: " << flops/((double)elapsed_ns.count()) << "\n" << std::endl;
  std::cout << "Gigabytes per second: " << bytes/((double)elapsed_ns.count()) << "\n" << std::endl;

  return 0;
}
