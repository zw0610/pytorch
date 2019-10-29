#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <vector>
#include <ctime>

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
  float* a = (float*)std::malloc(sizeof(float) * size);
  float* b = (float*)std::malloc(sizeof(float) * size);
  float* c = (float*)std::malloc(sizeof(float) * size);
  for (auto i = decltype(size){0}; i < size; ++i) {
    a[i] = ::rand(max_value);
    b[i] = ::rand(max_value);
    c[i] = ::rand(max_value);
  }

  for (auto i = decltype(warmup_iters){0}; i < warmup_iters; ++i) {
    for (auto i = decltype(size){0}; i < size; ++i) {
      c[i] += a[i] + b[i];
    }
  }

  const auto start = std::chrono::high_resolution_clock::now();

  for (auto i = decltype(iters){0}; i < iters; ++i) {
    for (auto i = decltype(size){0}; i < size; ++i) {
      c[i] += a[i] + b[i];
    }
  }

/* Loop assembly (from Godbolt)
.L11:
        movups  xmm0, XMMWORD PTR [rbp+0+rax]
        movups  xmm4, XMMWORD PTR [r12+rax]
        movups  xmm5, XMMWORD PTR [rbx+rax]
        addps   xmm0, xmm4
        addps   xmm0, xmm5
        movups  XMMWORD PTR [rbx+rax], xmm0
        add     rax, 16
        cmp     rax, r15
        jne     .L11
        mov     eax, edx
        cmp     esi, edx
        je      .L12
.L10:
        movsx   r9, eax
        lea     r10, [rbx+r9*4]
        movss   xmm0, DWORD PTR [r12+r9*4]
        addss   xmm0, DWORD PTR [rbp+0+r9*4]
        addss   xmm0, DWORD PTR [r10]
        lea     r9d, [rax+1]
        movss   DWORD PTR [r10], xmm0
        cmp     r9d, r13d
        jge     .L12
        movsx   r9, r9d
        add     eax, 2
        lea     r10, [rbx+r9*4]
        movss   xmm0, DWORD PTR [r12+r9*4]
        addss   xmm0, DWORD PTR [rbp+0+r9*4]
        addss   xmm0, DWORD PTR [r10]
        movss   DWORD PTR [r10], xmm0
        cmp     eax, r13d
        jge     .L12
        cdqe
        lea     r9, [rbx+rax*4]
        movss   xmm0, DWORD PTR [rbp+0+rax*4]
        addss   xmm0, DWORD PTR [r12+rax*4]
        addss   xmm0, DWORD PTR [r9]
        movss   DWORD PTR [r9], xmm0
.L12:
        sub     r8, 1
        jne     .L13
*/

  const auto end = std::chrono::high_resolution_clock::now();
  const auto elapsed = end - start;
  const auto elapsed_ns = std::chrono::duration_cast<ns>(elapsed);
  const std::size_t flops = 2 * size * iters;
  const std::size_t bytes = sizeof(float) * 3 * size * iters;

  std::cout << "Ran in " << elapsed_ns.count() << "ns\n" << std::endl;
  std::cout << "Array size: " << size << "\n" << std::endl;
  std::cout << "Each iteration took ~" << elapsed_ns.count()/iters << "ns\n" << std::endl;
  std::cout << "(perceived) Flops per nanonsecond: " << flops/((double)elapsed_ns.count()) << "\n" << std::endl;
  std::cout << "Gigabytes per second: " << bytes/((double)elapsed_ns.count()) << "\n" << std::endl;

  std::free(a);
  std::free(b);
  std::free(c);
  return 0;
}
