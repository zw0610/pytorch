#include <torch/torch.h>

#include <cstdint>
#include <chrono>
#include <iostream>

typedef std::chrono::nanoseconds ns;
constexpr uint64_t iters = 100000;

int main() {

  auto a = torch::ones({2, 3, 4});
  auto b = torch::ones({3, 4});

  const auto start = std::chrono::high_resolution_clock::now();

  for (auto i = decltype(iters){0}; i < iters; ++i) {
    auto c = a + b;
  }

  const auto end = std::chrono::high_resolution_clock::now();
  const auto elapsed = end - start;
  const auto elapsed_ns = std::chrono::duration_cast<ns>(elapsed);
  std::cout << "Ran in " << elapsed_ns.count() << "ns\n" << std::endl;
  std::cout << "Each iteration took ~" << elapsed_ns.count()/iters << "ns\n" << std::endl;
}
