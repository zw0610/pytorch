#include <torch/torch.h>

#include <array>
#include <cstdint>
#include <chrono>
#include <iostream>

typedef std::chrono::nanoseconds ns;
constexpr uint64_t iters = 100000;

int main() {

  std::array<int, 10> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::array<int, 10> b{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  const auto start = std::chrono::high_resolution_clock::now();

  for (auto i = decltype(iters){0}; i < iters; ++i) {

    for (auto i = decltype(a.size()){0}; i < a.size(); ++i) {
      std::array<int, 10> c{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
      c[i] = a[i] + b[i];
    }
  }

  const auto end = std::chrono::high_resolution_clock::now();
  const auto elapsed = end - start;
  const auto elapsed_ns = std::chrono::duration_cast<ns>(elapsed);
  std::cout << "Ran in " << elapsed_ns.count() << "ns\n" << std::endl;
  std::cout << "Each iteration took ~" << elapsed_ns.count()/iters << "ns\n" << std::endl;
}
