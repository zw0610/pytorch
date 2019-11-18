#include <torch/csrc/jit/fuser/cpu/interface.h>

#include <asmjit/asmjit.h>
#include <iostream>

namespace torch {
namespace jit {
namespace fuser {
namespace cpu {

// Signature of the generated function.
typedef int (*Func)(void);
typedef void (*addFunc)(unsigned, float*, float*, float*);

using namespace asmjit;
using namespace asmjit::x86;

// Returns true if the node is added to the fusion group, false o.w.
bool mergeNodeWithFusionGroup(const Node* const node, Node* fusion_group) {
  #if FUSER_DEBUG
    std::cout << "cpuMergeNodeWithFusionGroup" << std::endl;
  #endif // FUSER_DEBUG

  JitRuntime rt;

  CodeHolder code;
  code.init(rt.codeInfo());

  Assembler a(&code);

  // allocates sample arrays
  constexpr unsigned array_size = 32;
  float* out = (float*)malloc(sizeof(float) * array_size);
  float* lhs = (float*)malloc(sizeof(float) * array_size);
  float* rhs = (float*)malloc(sizeof(float) * array_size);

  // initializes sample arrays
  for (auto i = decltype(array_size){0}; i < array_size; ++i) {
    out[i] = 0.f;
    lhs[i] = 1.f;
    rhs[i] = 2.f;
  }

  // Creates fusion (element-by-element contiguous add)
  Label LoopInc = a.newLabel();
  Label LoopBody = a.newLabel();
  Label Exit = a.newLabel();

  // Short-circuits on size == 0
  a.test(edi, edi);
  a.je(Exit);

  // Stores # of loop iterations, sets loop counter to zero
  a.lea(r8d, dword_ptr(rdi, - 1)); // r8 = size - 1
  a.xor_(eax, eax); // clears eax
  a.jmp(LoopBody); // do () { } while () loop form

  // Loop incrementer
  a.bind(LoopInc);
  a.mov(rax, rdi); // offset = offset + 1

  // Loop body
  a.bind(LoopBody);
  a.vmovss(xmm0, dword_ptr(rdx, rax, 2)); // xmm0 = lhs[offset]
  a.vaddss(xmm0, xmm0, dword_ptr(rcx, rax, 2)); // xmm0 = xmm0 + rhs[offset]
  a.lea(rdi, dword_ptr(rax, 1)); // size = offset + 1
  a.vmovss(dword_ptr(rsi, rax, 2), xmm0); // out[offset] = xmm0

  // Checks if loop is finished
  a.cmp(rax, r8); // if offset == size - 1, terminate
  a.jne(LoopInc);

  // Exit
  a.bind(Exit);
  a.ret();

  // Jits the code and stores in function
  addFunc fn;
  Error err = rt.add(&fn, &code);
  if (err) {
    std::cout << "Error while jitting!" << std::endl;
  }

  // Computes the add
  fn(array_size, out, lhs, rhs);

  // Debug prints the output
  std::cout << "Jitted function result: ";

  for (auto i = decltype(array_size){0}; i < array_size; ++i) {
    std::cout << out[i] << " ";
  }
  std::cout << std::endl;

  // Frees sample arrays
  std::free(out);
  std::free(lhs);
  std::free(rhs);

  return false;
}

}}}} // namespace torch::jit::fuser::cpu
