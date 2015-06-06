/**
 * CUDA LineSearch Kernels (./cuda_sg/src/lib/lineSearch.cu)
 * Main system kernels for linesearch
 */

namespace gpu {

   /**
    * Discretize Line
    * @param N     Problem Size
    * @param D     Discrtization
    * @param x     X Vector
    * @param p     Direction Vector
    * @param h     Step Size
    * @param space Memory Space
    */
   __global__ void spcl( long int N , long int D ,  double* x, double* p, double h, double* space) {
      int i = blockDim.x * blockIdx.x + threadIdx.x;
      // int j = blockDim.y * blockIdx.y + threadIdx.y;
      int row = i / N;
      int col = i - row * N;

      //      printf("i : %d, row :%d , col %d \n", i, row, col );

      //  if (i < N && j < D) {
      space[row * N + col] = x[col] + p[col] * h * row;
      //}
   };

   /**
    * Line Evaluator
    * @param N        Porbelm Size
    * @param D        Discretixation
    * @param space    Memory Space
    * @param func_val Function Value
    */
   __global__ void fv( long int N , long int D , double* space, double* func_val) {
      int i = blockDim.x * blockIdx.x + threadIdx.x;

      //   if (i < D ) {
      double val = 0.0;
      FUNCTION(N, &space[i * N], &val);
      func_val[i] = val;
      //   };
   };
};

