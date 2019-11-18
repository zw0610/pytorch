#include <aten/src/ATen/core/jit_type.h>
#include <c10/core/DeviceType.h>

#include <torch/csrc/jit/fuser/interface.h>
#include <torch/csrc/jit/fuser/cpu/interface.h>

#include <torch/csrc/jit/fuser/compiler.h>
#include <torch/csrc/jit/fuser/executor.h>
#include <torch/csrc/jit/fuser/fallback.h>
#include <torch/csrc/jit/fuser/kernel_cache.h>

#include <asmjit/asmjit.h>

#include <stdexcept>

namespace torch {
namespace jit {

namespace detail {

// Note: CPU fusion is currently disabled due to test flakiness
bool cpu_fuser_enabled = false;

} // namespace detail

// TODO: make dbgs take stream to print on
namespace {

void printScalar(const Value* const value) {
  if (value->node()->kind() == prim::Constant) {
    std::cout << "Const Scalar: ";
  } else {
    std::cout << "Scalar: ";
  }

  if (value->type() == FloatType::get()) {
    std::cout << "float ";
    const float val = value->node()->f(attr::value);
    std::cout << val;
  } else if (value->type() == IntType::get()) {
    std::cout << "int ";
    const int val = value->node()->i(attr::value);
    std::cout << val;
  } else {
    std::cout << "unknown";
  }
  std::cout << std::endl;
}

void printStrides(const c10::VaryingStrides& strides) {
  std::cout << "Strides=";
  for (size_t i = 0; i < *(strides.size()); ++i) {
    std::cout << *(strides[i]);
    if(i != *(strides.size())-1)
      std::cout << ", ";
    else
      std::cout << ")";
  }

}

void printSizes(const c10::VaryingShape& sizes) {
  std::cout << "Sizes=( ";
  for (size_t i = 0; i < *(sizes.size()); ++i) {
    std::cout << *(sizes[i]);
    if(i != *(sizes.size())-1)
      std::cout << ", ";
    else
      std::cout << ")";
  }
}

void printCompleteTensor(const std::shared_ptr<c10::TensorType> tensor) {
  std::cout << "Complete Tensor: ";
  std::cout << *(tensor->device()) << " ";
  std::cout << *(tensor->scalarType()) << " ";
  std::cout << "nDims: " << *(tensor->dim()) << " ";
  std::cout << std::endl;
  printSizes(tensor->sizes());
  printStrides(tensor->strides());
}

void printValue(const Value* const value) {
  if (value->isCompleteTensor()) {
    printCompleteTensor(value->type()->expect<TensorType>());
  } else if (value->type()->isSubtypeOf(NumberType::get())) {
    printScalar(value);
  } else {
    std::cout << "Request to print unknown value" << std::endl;
  }
}

// Returns true if and only if value is a scalar or a complete tensor type
bool validateValue(const Value* const value, const bool dbg = false) {
  if (dbg) {
    printValue(value);
  }

  if (value->isCompleteTensor() || value->type()->isSubtypeOf(NumberType::get())) {
    return true;
  }

  return false;
}

// Returns true if all inputs and outputs are complete tensors or scalars
// Note: complete tensor means device, nDims, sizes, and strides are known
// In particular, all optional values of sizes and strides have values
bool validateNode(const Node* const node, const bool dbg = false) {
  auto inputs = node->inputs();
  auto outputs = node->outputs();

  if (dbg) {
    std::cout << "nInputs: " << inputs.size() << std::endl;
    std::cout << "nOutputs: " << outputs.size() << std::endl;
  }

  if (dbg) {
    std::cout << "Inputs: " << std::endl;
  }

  for (auto i = decltype(inputs.size()){0}; i < inputs.size(); ++i) {
    if (!validateValue(inputs[i], dbg)) {
      return false;
    }
  }

  if (dbg) {
    std::cout << "Outputs: " << std::endl;
  }

  for (auto i = decltype(outputs.size()){0}; i < outputs.size(); ++i) {
    if (!validateValue(outputs[i], dbg)) {
      return false;
    }
  }

  return true;
}

} // namespace

std::vector<bool> canCollapseDimsDown(const std::shared_ptr<c10::TensorType> tensor){
  int64_t ndims = *(tensor->dim());

  //Flags to see if the current dim can be fused with the one after
  //Goes left to right, furthest right doesn't need a flag
  std::vector<bool> canCollapseDown(ndims, true);

  for (int64_t d = 0; d < ndims - 1; d++) {
    int64_t stride = *(tensor->strides()[d]);
    int64_t stride_p_1 = *(tensor->strides()[d+1]);
    int64_t size_p_1 = *(tensor->sizes()[d+1]);

    if( (stride_p_1 * size_p_1 != stride)
	&& !(stride_p_1 == 0 && stride == 0) )
      canCollapseDown[d] = false;

  }

  canCollapseDown[ndims-1] = true;

  return canCollapseDown;
}

bool cudaMergeNodeWithFusionGroup(const Node* const node, Node* fusion_group) {
  #if FUSER_DEBUG
    std::cout << "cudaMergeNodeWithFusionGroup" << std::endl;
  #endif // FUSER_DEBUG

  bool dbg = FUSER_DEBUG;

  int64_t ndims = *(node->inputs()[0]->type()->expect<TensorType>()->dim());
  std::vector< std::vector<bool> > collapse_vecs;


  //Check how we could dimensionally reduce each input
  for(const auto& value : node->inputs())
    if(value->isCompleteTensor()){
      assert(*(value->type()->expect<TensorType>()->dim()) == ndims);
      collapse_vecs.push_back(canCollapseDimsDown(value->type()->expect<TensorType>()));
    }

  //Check how we could dimennsionally reduce each output
  for(const auto& value : node->outputs())
    if(value->isCompleteTensor()){
      assert(*(value->type()->expect<TensorType>()->dim()) == ndims);
      collapse_vecs.push_back(canCollapseDimsDown(value->type()->expect<TensorType>()));
    }

  std::vector<bool> dim_collapse = collapse_vecs[0];

  for(auto it = collapse_vecs.begin() + 1; it!=collapse_vecs.end(); ++it){
    for(int64_t d = 0; d<ndims; d++){
      dim_collapse[d] = dim_collapse[d] && (*it)[d];
    }
  }

  //Contig not the right word here because the tensor:
  //Size(4, 4, 2) stride(16, 4, 2) will be fully
  //collapsable but not contiguous
  bool contig = true;
  for(const auto iscontig : dim_collapse)
    contig = contig && iscontig;

  if(contig)
    std::cout<<"All tensors are contiguous"<<std::endl;

  bool first = true;
  for (auto i = decltype(dim_collapse.size()){0}; i < dim_collapse.size() - 1 ; ++i) {
    if(dim_collapse[i]){
      if(first){
	std::cout<<"Tensors could be collapsed on Dims = ("<<i;
	first = false;
      }else{
	std::cout<<", "<<i;
      }
    }
  }
  if(!first) std::cout<<")"<<std::endl;


  if(node->kind() ==  aten::add){
    std::cout<<"Can fuse node!"<<std::endl;
    return true;
  }


  return false;
}

// Given a node and a fusion group, returns true if the node can be
// merged into the fusion group and false if it cannot.
// The fusion_group may be empty, specified by a nullptr.
// TODO: make this actually work (currently always returns false)
// TODO: validate that all inputs and outputs are on the same device (no multi-device fusion)

bool mergeNodeWithFusionGroup(const Node* const node, Node* fusion_group) {
  #if FUSER_DEBUG
    std::cout << "interface.cpp: mergeNodeWithFusionGroup()" << std::endl;
    const auto is_valid = validateNode(node, true);
    std::cout << "is_valid: " << is_valid << std::endl;
  #else
    const auto is_valid = validateNode(node, false);
  #endif // FUSER_DEBUG

  if (!is_valid) {
    return false;
  }

  const auto& outputs = node->outputs();

  // Only single-output fusions (for now)
  if (outputs.size() != 1) {
    return false;
  }

  const auto& output = node->output();

  // Only tensor output fusions (for now)
  if (!(output->isCompleteTensor())) {
    return false;
  }

  const std::shared_ptr<c10::TensorType> out_tensor = output->type()->expect<TensorType>();
  const auto fusion_device = *(out_tensor->device());

  #if FUSER_DEBUG
  std::cout << "Fusion Device: " << fusion_device<<std::endl;
  #endif // FUSER_DEBUG

  if (fusion_device.type() == c10::kCPU) {
    #if FUSER_DEBUG
      std::cout << "Fusing on CPU" << std::endl;
    #endif // FUSER_DEBUG
    return torch::jit::fuser::cpu::mergeNodeWithFusionGroup(node, fusion_group);
  } else if (fusion_device.type() == c10::kCUDA) {
    #if FUSER_DEBUG
      std::cout << "Fusing on CUDA" << std::endl;
    #endif // FUSER_DEBUG
    return cudaMergeNodeWithFusionGroup(node, fusion_group);
  } else {
    std::cout << "unknown fusion device: " << fusion_device << std::endl;
    return false;
  }

  return false;
}


// OLD STUFF BELOW HERE























int64_t registerFusion(const Node* fusion_group) {
  return fuser::registerFusion(fusion_group);
}

void runFusion(const int64_t key, Stack& stack) {
  const auto result = fuser::runFusion(key, stack);
  if (!result)
    fuser::runFallback(key, stack);
}

bool canFuseOnCPU() {
  return fuser::hasFusionBackend(at::DeviceType::CPU) &&
      detail::cpu_fuser_enabled;
}

bool canFuseOnGPU() {
  return fuser::hasFusionBackend(at::DeviceType::CUDA);
}

void overrideCanFuseOnCPU(bool value) {
  detail::cpu_fuser_enabled = value;
}

// Uses the above interface by stuffing the graph into a node and treating that
// node as a fusion group.
std::vector<at::Tensor> debugLaunchGraph(
    Graph& graph,
    at::ArrayRef<at::Tensor> inputs) {
  // Creates a fusion group node
  auto wrapper_graph = std::make_shared<Graph>();
  Node* fusion_group = wrapper_graph->insertNode(
      wrapper_graph->createWithSubgraph(prim::FusionGroup));
  fusion_group->g_(attr::Subgraph, graph.copy());
  for (size_t i = 0; i < graph.inputs().size(); ++i) {
    fusion_group->addInput(wrapper_graph->addInput());
  }
  for (size_t i = 0; i < graph.outputs().size(); ++i) {
    wrapper_graph->registerOutput(fusion_group->addOutput());
  }

  // Creates the stack, registers and runs the fusion
  Stack stack = fmap<IValue>(inputs);
  const auto key = fuser::registerFusion(fusion_group);
  fuser::runFusion(key, stack);
  return fmap(stack, [](const IValue& iv) { return iv.toTensor(); });
}

std::string debugGetFusedKernelCode(
    Graph& graph,
    at::ArrayRef<at::Tensor> inputs) {
  // Creates a fusion group node
  auto wrapper_graph = std::make_shared<Graph>();
  Node* fusion_group =
      wrapper_graph->insertNode(wrapper_graph->createWithSubgraph(prim::FusionGroup));
  fusion_group->g_(attr::Subgraph, graph.copy());
  for (size_t i = 0; i < graph.inputs().size(); ++i) {
    fusion_group->addInput(wrapper_graph->addInput());
  }
  for (size_t i = 0; i < graph.outputs().size(); ++i) {
    wrapper_graph->registerOutput(fusion_group->addOutput());
  }

  // Creates the stack, registers and runs the fusion
  Stack stack = fmap<IValue>(inputs);
  const auto key = fuser::registerFusion(fusion_group);

  std::string code;
  if (!fuser::runFusion(key, stack, &code)) {
    throw std::runtime_error("Could not run fusion for graph");
  }

  return code;
}

size_t nCompiledKernels() {
  return fuser::nCompiledKernels();
}

} // namespace jit
} // namespace torch
