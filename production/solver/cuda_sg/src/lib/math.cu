namespace gpu {

   __global__ void lineDiscretize( long int N , long int D ,  double* x, double* p, double h, double* space) {
      int i = blockDim.x * blockIdx.x + threadIdx.x;
      int j = blockDim.y * blockIdx.y + threadIdx.y;

      if (i < N && j < D) {
         space[j * N + i] = x[i] + p[i] * h * j;
         printf("i %d ,  j %d ,  value : %f \n", i, j, space[j * N + i]);
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

