#include <cstdint>
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <string>
#include <memory>
#include <vector>
#include <utility>
#include <iostream>
#include <exception>

typedef std::chrono::nanoseconds ns;
constexpr uint64_t warmup_iters = 1000;
constexpr uint64_t iters = 500000;
constexpr int max_value = 50;

namespace jabberwocky {

/*
TODO: refactor benchmark tests (pass a lambda to a benchmark runner)
TODO: compare benchmark tests with directly invoking the kernel
TODO: sizes + strides in kernel
TODO: cachegrind?
TODO: shared_ptr vs intrusive_ptr perf
*/

static constexpr unsigned MAX_TENSOR_DIMS = 5;
using SizesType = std::array<std::size_t, MAX_TENSOR_DIMS>;
using StridesType = SizesType;
static constexpr SizesType size_and_stride_init{0, 0, 0, 0, 0};

struct Tensor {

  // Creates a 1D tensor with the specified number of elements
  Tensor(
    const std::size_t _numel)
  : numel_{_numel} {
    nDims_ = 1;
    sizes_[0] = numel_;
    strides_[0] = 1;
    allocate_storage();
  }

  void allocate_storage() {
    const auto size = sizeof(float) * numel_;
    storage_ = (float*)std::malloc(size);
  }

  ~Tensor() {
    // Note: calling std::free on nullptr is fine
    std::free(storage_);
  }

  // Copy constructor and copy assignment operator (deleted)
  Tensor(const Tensor& other) = delete;
  Tensor& operator=(const Tensor& other) = delete;

  // Move constructor and move assignment operator
  Tensor(Tensor&& other) {
    swap(std::move(other));
  }
  Tensor& operator=(Tensor&& other) {
    swap(std::move(other));
    return *this;
  }
  void swap(Tensor&& other) {
    std::swap(numel_, other.numel_);
    std::swap(nDims_, other.nDims_);
    std::swap(sizes_, other.sizes_);
    std::swap(strides_, other.strides_);
    std::swap(storage_, other.storage_);
    std::swap(p_t_, other.p_t_);
  }

  std::size_t numel() const { return numel_; }
  float* const storage() {
    if (p_t_) {
      return p_t_->storage_;
    }
    return storage_;
  }
  const float* const storage() const {
    if (p_t_) {
      return p_t_->storage_;
    }
    return storage_;
  }

  // Atomic refcounting operations
  void acquire() {
    ++ref_count_;
  }
  bool release() {
    if (ref_count_-- == 0) {
      return true;
    }

    return false;
  }

  std::size_t numel_ = 0;
  std::size_t nDims_ = 0;
  SizesType sizes_ = size_and_stride_init;
  StridesType strides_ = size_and_stride_init;
  float* storage_ = nullptr;
  Tensor* p_t_ = nullptr;
  std::size_t ref_count_{0};
};

struct TensorFreeList {
  void push(Tensor* const t) {
    t->p_t_ = head_;
    head_ = t;
  }

  Tensor* pop() {
    Tensor* const t = head_;
    if (t != nullptr) {
      head_ = t->p_t_;
    }

    return t;
  }

private:
  Tensor* head_ = nullptr;
};

static TensorFreeList* tfl = new TensorFreeList{};

struct TensorRef {
  TensorRef(const std::size_t numel) {
    t_ = tfl->pop();
    if (t_ == nullptr) {
      t_ = new Tensor{numel};
    } else {
      t_->~Tensor();
      t_ = new(t_) Tensor{numel};
    }
  }
  ~TensorRef() {
    if (t_->release()) {
      tfl->push(t_);
    }
  }

  // Copy constructor and copy assignment operator
  TensorRef(const TensorRef& other) {
    copy(other);
  }
  TensorRef& operator=(const TensorRef& other) {
    t_->release();
    copy(other);
    return *this;
  }
  void copy(const TensorRef& other) {
    other.t_->acquire();
    t_ = other.t_;
  }

  // Move constructor and move assignment operator
  TensorRef(TensorRef&& other) {
    swap(std::move(other));
  }
  TensorRef& operator=(TensorRef&& other) {
    swap(std::move(other));
    return *this;
  }
  void swap(TensorRef&& other) {
    std::swap(t_, other.t_);
  }

  std::size_t numel() const { return t_->numel(); }
  float* const storage() { return t_->storage(); }
  const float* const storage() const { return t_->storage(); }
  Tensor* const borrow() { return t_; }
  const Tensor* const borrow() const { return t_; }

private:
  // Note: this should never be nullptr
  Tensor* t_;
};

bool check_shapes(const TensorRef& lhs, const TensorRef& rhs) {
  if (lhs.numel() != rhs.numel()) {
    return false;
  }

  return true;
}

TensorRef full(const float f, const std::size_t numel) {
  TensorRef t{numel};
  float* const storage = t.storage();

  for (auto i = decltype(t.numel()){0}; i < t.numel(); ++i) {
    storage[i] = f;
  }

  return t;
}

void add_unchecked(
  float* const out
, const SizesType& out_sizes
, const StridesType& out_strides
, const float* const lhs
, const SizesType& lhs_sizes
, const StridesType& lhs_strides
, const float* const rhs
, const SizesType& rhs_sizes
, const StridesType& rhs_strides
) {
  return;
}

void add_unchecked(
  const std::size_t numel
, float* const out
, const float* const lhs
, const float* const rhs) {

  for (auto i = decltype(numel){0}; i < numel; ++i) {
    out[i] = lhs[i] + rhs[i];
  }
}

TensorRef add(const TensorRef& lhs, const TensorRef& rhs) {
  // const auto has_valid_shapes = check_shapes(lhs, rhs);
  TensorRef out{lhs.numel()};
  add_unchecked(lhs.numel(), out.storage(), lhs.storage(), rhs.storage());
  return out;
}

// void print(const Tensor& t) {
//   std::cout << "Tensor : ";

//   for (auto i = decltype(t.numel()){0}; i < t.numel(); ++i) {
//     std::cout << t.storage()[i] << " " << std::endl;
//   }
// }

namespace test {

void ASSERT(const bool asserted, const std::string s) {
  if (!asserted) {
    std::cerr << s << std::endl;
  }
}

void test_tensor_free_list() {
  Tensor* t0 = new Tensor{1};
  Tensor* t1 = new Tensor{1};

  TensorFreeList list{};

  list.push(t0);
  auto* head0 = list.pop();
  ASSERT(head0 == t0, "Free list push/pop failure.");
  list.push(t0);
  list.push(t1);
  auto* head1 = list.pop();
  auto* head2 = list.pop();
  auto* head3 = list.pop();
  auto* head4 = list.pop();
  ASSERT(head1 == t1, "Free list failed to return in order.");
  ASSERT(head2 == t0, "Free list failed to return all enqueud tensors.");
  ASSERT(head3 == head4 && head4 == nullptr, "Free list failed to return nullptr.");
}

void test_refs() {
  auto t = full(1.f, 5);
  TensorRef v{t};

  ASSERT(v.storage()[0] == t.storage()[0], "View element not equal to tensor it was constructed from");
  t.storage()[0] = 2.f;
  ASSERT(v.storage()[0] == t.storage()[0], "Tensor changes not propagated to view.");
  v.storage()[0] = 3.f;
  ASSERT(v.storage()[0] == t.storage()[0], "View changes not propagated.");
}

void run_tests() {
  test_refs();
  test_tensor_free_list();

  std::cerr << "Ran tests successfully." << std::endl;
}

} // test
} // jabberwocky

int rand(const int max_value) {
  return 1 + std::rand()/((RAND_MAX + 1u)/max_value);
}

int main() {
  std::srand(std::time(0));
  const auto size = ::rand(max_value);

  jabberwocky::test::run_tests();

  jabberwocky::TensorRef a = jabberwocky::full(1.f, size);
  jabberwocky::TensorRef b = jabberwocky::full(2.f, size);
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
