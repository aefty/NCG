namespace cuda {

   __device__ float atomicMax(float* address, float val) {
      int* address_as_int = (int*)address;
      int old = *address_as_int, assumed;

      while (val > __int_as_float(old)) {
         assumed = old;
         old = atomicCAS(address_as_int, assumed,
                         __float_as_int(val));
      }

      return __int_as_float(old);
   }

   __global__ void lineDiscretize( long int N , long int D ,  double* x, double* p, double h, double* space) {
      int i = blockDim.x * blockIdx.x + threadIdx.x;
      int j = blockDim.y * blockIdx.y + threadIdx.y;

      if (i < N && j < D) {
         space[j * N + i] = x[i] + p[i] * h * j;
      }
   };

   __global__ void lineValue( long int N , long int D , double* space, double* func_val) {
      int i = blockDim.x * blockIdx.x + threadIdx.x;

      if (i < D ) {
         double val = 0.0;
         FUNCTION(N, &space[i * N], &val);
         func_val[i] = val;
      };
   };
};

