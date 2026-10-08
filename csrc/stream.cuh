#pragma once

#include <ATen/cuda/CUDAContext.h>

// PyTorch's current CUDA stream on the current device. Kernels launched there follow PyTorch's stream ordering and can be
// captured in a CUDA graph; the legacy default stream can do neither.
inline cudaStream_t sage_current_stream() {
  return at::cuda::getCurrentCUDAStream().stream();
}
