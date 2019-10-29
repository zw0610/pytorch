#include <cstdint>
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <string>
#include <memory>
#include <vector>

namespace jabberwocky {

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

std::shared_ptr<Storage> get_storage(const std::size_t size) {
  return std::make_shared<Storage>(true, size);
}

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

struct Tensor {

  Tensor(
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

Tensor full(const float f, const std::vector<int>& sizes) {
  const auto numel = compute_numel(sizes);
  return Tensor(numel, sizes);
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

typedef std::chrono::nanoseconds ns;
constexpr uint64_t warmup_iters = 1000;
constexpr uint64_t iters = 200000;
constexpr int max_value = 50;

int rand(const int max_value) {
  return 1 + std::rand()/((RAND_MAX + 1u)/max_value);
}

int main() {
  std::srand(std::time(0));
  const auto size = ::rand(max_value);

  jabberwocky::Tensor a = jabberwocky::full(1.f, {size});
  jabberwocky::Tensor b = jabberwocky::full(2.f, {size});

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
