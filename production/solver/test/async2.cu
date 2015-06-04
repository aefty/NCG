// Copyright 2012 NVIDIA Corporation

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result) {
  #if defined(DEBUG) || defined(_DEBUG)

  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }

  #endif
  return result;
}

__global__ void kernel(double* a, int offset) {
  int i = offset + threadIdx.x + blockIdx.x * blockDim.x;
  double x = (double)i;
  double s = sinf(x);
  double c = cosf(x);
  a[i] = a[i] + sqrtf(s * s + c * c);
}

double maxError(double* a, int n) {
  double maxE = 0;

  for (int i = 0; i < n; i++) {
    double error = fabs(a[i] - 1.0);

    if (error > maxE) { maxE = error; }
  }

  return maxE;
}

int main(int argc, char** argv) {
  const int n = 4 * 1024;
  const int blockSize = 256, nStreams = n;
  const int streamSize = n / nStreams;
  const int streamBytes = streamSize * sizeof(double);
  const int bytes = n * sizeof(double);

  int devId = 0;

  if (argc > 1) { devId = atoi(argv[1]); }

  cudaDeviceProp prop;
  checkCuda( cudaGetDeviceProperties(&prop, devId));
  printf("Device : %s\n", prop.name);
  checkCuda( cudaSetDevice(devId) );

  // allocate pinned host memory and device memory
  double* a, *d_a;
  checkCuda( cudaMallocHost((void**)&a, bytes) );      // host pinned
  checkCuda( cudaMalloc((void**)&d_a, bytes) ); // device

  float ms; // elapsed time in milliseconds

  // create events and streams
  cudaEvent_t startEvent, stopEvent;
  cudaStream_t stream[nStreams];
  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );


  // asynchronous version 1: loop over {copy, kernel, copy}
  memset(a, 0, bytes);
  checkCuda( cudaEventRecord(startEvent, 0) );

  for (int i = 0; i < nStreams; ++i) {

    checkCuda( cudaStreamCreate(&stream[i]) );

    int offset = 0;//i * streamSize;

    checkCuda( cudaMemcpyAsync(&d_a[offset], &a[offset],
                               streamBytes, cudaMemcpyHostToDevice,
                               stream[i]) );
    kernel <<< streamSize / blockSize, blockSize, 0, stream[i] >>> (d_a, offset);
    checkCuda( cudaMemcpyAsync(&a[offset], &d_a[offset],
                               streamBytes, cudaMemcpyDeviceToHost,
                               stream[i]) );
  }

  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  printf("Time for asynchronous V1 transfer and execute (ms): %f\n", ms);
  printf("  max error: %e\n", maxError(a, n));



  // cleanup
  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );


  for (int i = 0; i < nStreams; ++i) {
    checkCuda( cudaStreamDestroy(stream[i]) );
  }

  cudaFree(d_a);
  cudaFreeHost(a);

  return 0;
}