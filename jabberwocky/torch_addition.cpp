#include <torch/torch.h>

#include <cstdint>
#include <chrono>
#include <iostream>

typedef std::chrono::nanoseconds ns;
constexpr uint64_t warmup_iters = 500000;
constexpr uint64_t iters = 5000000;
constexpr int max_value = 50;

int rand(const int max_value) {
  return 1 + std::rand()/((RAND_MAX + 1u)/max_value);
}

int main() {

  std::srand(std::time(0));
  const auto size = ::rand(max_value);

  auto a = torch::ones({size});
  auto b = torch::ones({size});

  for (auto i = decltype(warmup_iters){0}; i < warmup_iters; ++i) {
    auto c = a + b;
  }

  const auto start = std::chrono::high_resolution_clock::now();

  for (auto i = decltype(iters){0}; i < iters; ++i) {
    auto c = a + b;
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
}
