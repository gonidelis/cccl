/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include <cub/device/device_segmented_reduce.cuh>

#include <thrust/device_vector.h>

#include <climits>
#include <cstddef>

#include "catch2_test_helper.h"
<<<<<<< HEAD
#include "thrust/detail/raw_pointer_cast.h"
=======
>>>>>>> 3aae62ed4206f0e6963279fd6d6e9eb5629e8dc6

struct CustomMin
{
  template <typename T>
  __device__ __forceinline__ T operator()(const T& a, const T& b) const
  {
    return (b < a) ? b : a;
  }
};

CUB_TEST("Device segmented reduce works with int data elements", "[segmented_reduce][device]")
{
  // example-begin segmented-reduce-reduce
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 3, 3, 7};
<<<<<<< HEAD
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  thrust::device_vector<int> d_out(3);
=======
  thrust::device_vector<int> d_in{1};
  thrust::device_vector<int> d_out;
>>>>>>> 3aae62ed4206f0e6963279fd6d6e9eb5629e8dc6
  CustomMin min_op;
  int initial_value{INT_MAX};

  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Reduce(
<<<<<<< HEAD
    d_temp_storage,
    temp_storage_bytes,
    d_in.begin(),
    d_out.begin(),
    num_segments,
    d_offsets.begin(),
    d_offsets.begin() + 1,
    min_op,
    initial_value);

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // Run reduction
  cub::DeviceSegmentedReduce::Reduce(
    d_temp_storage,
    temp_storage_bytes,
    d_in.begin(),
    d_out.data(),
    num_segments,
    d_offsets.begin(),
    d_offsets.begin() + 1,
    min_op,
    initial_value);

  thrust::device_vector<int> expected{6, INT_MAX, 0};
  // example-end segmented-reduce-reduce

  REQUIRE(d_out == expected);
=======
    d_temp_storage, temp_storage_bytes, d_in.data(), d_out.data(), num_segments, d_offsets.data(), d_offsets.data() + 1, min_op, initial_value);
  
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  
  // Run reduction
  cub::DeviceSegmentedReduce::Reduce(
    d_temp_storage, temp_storage_bytes, d_in.data(), d_out.data(), num_segments, d_offsets.data(), d_offsets.data() + 1, min_op, initial_value);

  // example-end segmented-reduce-reduce

  REQUIRE(d_out == thrust::device_vector<int>{6, INT_MAX, 0});
>>>>>>> 3aae62ed4206f0e6963279fd6d6e9eb5629e8dc6
}
