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

#include <cub/block/block_scan.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/std/numeric>

#include <algorithm>

#include "catch2_test_helper.h"

thrust::host_vector<int> host_scan(thrust::host_vector<int> h_in, int initial_value)
{
  cuda::std::inclusive_scan(h_in.begin(), h_in.end(), h_in.begin(), thrust::max<int>, initial_value);
  return h_in;
}

// example-begin inclusive-scan-init-value
__global__ void InclusiveScanKernel(int* input, int* output, int initial_value = 1)
{
  // Specialize BlockScan for a 1D block of 128 threads of type int
  typedef cub::BlockScan<int, 128> BlockScan;

  // Allocate shared memory for BlockScan
  __shared__ typename BlockScan::TempStorage temp_storage;

  // Obtain input item for each thread
  int thread_data = input[threadIdx.x];

  // Collectively compute the block-wide inclusive prefix max scan
  BlockScan(temp_storage).InclusiveScan(thread_data, thread_data, initial_value, cub::Max());

  output[threadIdx.x] = thread_data;
}
// example-end inclusive-scan-init-value

CUB_TEST("Block value-based inclusive scan works with initial value", "[block]")
{
  thrust::host_vector<int> h_in(128);
  int value = 0;
  thrust::generate(h_in.begin(), h_in.end(), [&value]() {
    return (value % 2 == 0) ? value++ : -value++;
  });

  thrust::device_vector<int> d_in = h_in;
  thrust::device_vector<int> d_out(128);
  int initial_value = 1;

  InclusiveScanKernel<<<1, 128>>>(
    thrust::raw_pointer_cast(d_in.data()), thrust::raw_pointer_cast(d_out.data()), initial_value);
  cudaDeviceSynchronize();

  thrust::host_vector<int> h_out = host_scan(h_in, initial_value);

  REQUIRE(h_out == d_out);
}

// example-begin inclusive-scan-init-value-aggregate
__global__ void InclusiveScanKernelAggregate(int* input, int* output, int* block_aggregate, int initial_value = 1)
{
  // Specialize BlockScan for a 1D block of 128 threads of type int
  typedef cub::BlockScan<int, 128> BlockScan;

  // Allocate shared memory for BlockScan
  __shared__ typename BlockScan::TempStorage temp_storage;

  // Obtain input item for each thread
  int thread_data = input[threadIdx.x];

  // Collectively compute the block-wide inclusive prefix max scan
  BlockScan(temp_storage).InclusiveScan(thread_data, thread_data, initial_value, cub::Max(), *block_aggregate);

  output[threadIdx.x] = thread_data;
}
// example-end inclusive-scan-init-value-aggregate

CUB_TEST("Block value-based inclusive scan aggregate works with initial value", "[block]")
{
  thrust::host_vector<int> h_in(128);
  int value = 0;
  thrust::generate(h_in.begin(), h_in.end(), [&value]() {
    return (value % 2 == 0) ? value++ : -value++;
  });

  thrust::device_vector<int> d_in = h_in;
  thrust::device_vector<int> d_out(h_in.size());
  int initial_value = 1;
  thrust::device_vector<int> block_aggregate(1);

  InclusiveScanKernelAggregate<<<1, 128>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    thrust::raw_pointer_cast(block_aggregate.data()),
    initial_value);
  cudaDeviceSynchronize();

  thrust::host_vector<int> h_out = host_scan(h_in, initial_value);

  int expected_aggregate = *std::max_element(h_out.begin(), h_out.end());

  REQUIRE(expected_aggregate == block_aggregate[0]);
  REQUIRE(h_out == d_out);
}
