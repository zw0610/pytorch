#include <algorithm>
#include <iostream>

#include <ATen/ATen.h>
#include <ATen/native/Repeat.h>
#include <ATen/Parallel.h>


static void compute_cpu(int64_t *repeat_ptr, int64_t *cumsum_ptr, int64_t *result_ptr, int64_t size) {
    at::parallel_for(0, size, 1, [&](int64_t i_begin, int64_t i_end) {
        for(int64_t i = i_begin; i < i_end; i++) {
            int64_t end = cumsum_ptr[i];
            int64_t size = repeat_ptr[i];
            int64_t start = end - size;
            for(int64_t j = start; j < end; j++) {
                result_ptr[j] = i;
            }
        }
    });
}

static inline IntArrayRef to_intarrayref(const Tensor& t){
    TORCH_CHECK(t.dim() == 1, "shape tensor should be a vector");
    TORCH_CHECK(t.scalar_type() == at::kLong, "shape has to be Long tensor");
    return IntArrayRef(t.data_ptr<int64_t>(), t.size(0));
}

namespace at { namespace native {

Tensor naive_loop(const Tensor& input, const Tensor& repeats) {
  TORCH_CHECK(repeats.dim() == 1, "repeat_interleave only accept 1D vector as repeat");
  TORCH_CHECK(repeats.scalar_type() == at::kLong, "repeats has to be Long tensor");
  TORCH_CHECK((repeats >= 0).all().item<uint8_t>(), "repeats can not be negative");

  Tensor repeats_ = repeats.contiguous();
  auto repeats_total = *(repeats_.sum(0).data_ptr<int64_t>());

  auto output = at::empty({repeats_total}, input.options());
  int64_t* repeats_ptr = repeats_.data_ptr<int64_t>();
  auto input_ptr = static_cast<char*>(input.data_ptr());
  auto output_ptr = static_cast<char*>(output.data_ptr());
  const auto element_size = elementSize(input.scalar_type());
  for (auto i = 0; i < repeats_.size(0); ++i) {
    for (auto j = 0; j < *(repeats_ptr + i); j++) {
      memcpy(output_ptr, input_ptr, element_size);
      output_ptr += element_size;
    }
    input_ptr += element_size;
  }
  return output;
}

template <unsigned N>
Tensor sharded_loop(const Tensor& input, const Tensor& repeats) {
  TORCH_CHECK(repeats.dim() == 1, "repeat_interleave only accept 1D vector as repeat");
  TORCH_CHECK(repeats.scalar_type() == at::kLong, "repeats has to be Long tensor");
  TORCH_CHECK((repeats >= 0).all().item<uint8_t>(), "repeats can not be negative");

  Tensor repeats_ = repeats.contiguous();
  auto repeats_total = *(repeats.sum(0).data<int64_t>());

  auto n = repeats_.size(0);

  auto repeats_cumsum = at::empty({n}, repeats_.options());
  int64_t* repeats_ptr = repeats_.data_ptr<int64_t>();
  int64_t* repeats_cumsum_ptr = repeats_cumsum.data_ptr<int64_t>();

  auto num_shards =
      (repeats_total / N + (repeats_total % N != 0));
  auto shard_bounds = at::zeros({num_shards}, repeats_.options());
  int64_t* shard_bounds_ptr = shard_bounds.data_ptr<int64_t>();
  int64_t current_shard{0};

  int64_t total{0};
  int64_t shard_capacity = N;
  for (int64_t i = 0; i < n; ++i) {
    auto current_el = *(repeats_ptr + i);
    total += current_el;
    *(repeats_cumsum_ptr + i) = total;

    if (C10_UNLIKELY(current_el > shard_capacity)) {
      while (current_el > shard_capacity) {
        current_el -= shard_capacity;
        shard_capacity = N;
        ++current_shard;
        *(shard_bounds_ptr + current_shard) = i;
      }
    }

    shard_capacity -= current_el;
  }

  auto output = at::empty({repeats_total}, input.options());
  auto output_ptr = static_cast<char*>(output.data_ptr());
  auto input_ptr = static_cast<char*>(input.data_ptr());
  const auto element_size = elementSize(input.scalar_type());

  auto f = [output_ptr,
            shard_bounds_ptr,
            input_ptr,
            repeats_ptr,
            repeats_cumsum_ptr,
            repeats_total,
            element_size](int64_t i_begin, int64_t i_end) {
    for (auto shard_no = i_begin; shard_no < i_end; ++shard_no) {
      auto offset = *(shard_bounds_ptr + shard_no);
      auto source = input_ptr + offset * element_size;
      auto dest = output_ptr + shard_no * N * element_size;
      auto local_repeats = repeats_ptr + offset;

      auto remaining = *(repeats_cumsum_ptr + offset) - shard_no * N;
      auto total_writes =
          std::min<int64_t>(N, repeats_total - shard_no * N);
      for (auto i = 0; i < total_writes; i++) {
        memcpy(dest, source, element_size);
        dest += element_size;
        --remaining;
        if (remaining == 0) {
          source += element_size;
          ++local_repeats;
          remaining = *local_repeats;
        }
      }
    }
  };

  at::parallel_for(0, num_shards, 1, f);
  return output;
}

Tensor repeat_interleave_cpu(const Tensor &repeat) {
    return repeat_interleave_common<compute_cpu>(repeat);
}

Tensor repeat_interleave(
    const Tensor& self,
    const Tensor& repeats,
    c10::optional<int64_t> dim,
    c10::optional<int64_t> impl) {
  Tensor input = self;
  if (!dim) {
    input = self.flatten();
    dim = 0;
  }

  Tensor repeats_ = repeats;
  if (repeats.dim() == 0 || (repeats.dim() == 1 && repeats.size(0) == 1)) {
    repeats_ = repeats.reshape({1}).expand({input.size(dim.value())});
  } else if (repeats.dim() == 1) {
    TORCH_CHECK(
        repeats.size(0) == input.size(dim.value()),
        "repeats must have the same size as input along dim")
  } else {
    AT_ERROR("repeats must be 0-dim or 1-dim tensor");
  }

  // Experiment with different values. Not for submission.
  switch (impl.value_or(0)) {
    case 0:
      return input.index_select(dim.value(), at::repeat_interleave(repeats_));
    case 1:
        {
            Tensor indices = at::repeat_interleave(repeats_);

            Tensor indices_sizes = at::ones({self.dim()}, TensorOptions(at::kLong));
            indices_sizes[dim.value()] = indices.size(0);
            indices = indices.reshape(to_intarrayref(indices_sizes));

            Tensor expanded_indices_shape = at::_shape_as_tensor(self).clone();
            expanded_indices_shape[dim.value()] = -1;
            indices = indices.expand(to_intarrayref(expanded_indices_shape));
            return input.gather(dim.value(), indices);
        }
      
    // Just the index generation to easily measure that portion.
    case 100:
      return at::repeat_interleave(repeats_);

    // NOTE: prototypes assume dim=0.
    case 2:
      return naive_loop(input, repeats_);
    case 3:
      return sharded_loop<1024>(input, repeats_);
    case 4:
      return sharded_loop<2048>(input, repeats_);
    default:
      return sharded_loop<4019>(input, repeats_);
  }
}

Tensor repeat_interleave(
    const Tensor& self,
    int64_t repeats,
    c10::optional<int64_t> dim,
    c10::optional<int64_t> impl) {
  return at::native::repeat_interleave(
      self, at::tensor({repeats}, self.options().dtype(kLong)), dim, impl);
}
}}
