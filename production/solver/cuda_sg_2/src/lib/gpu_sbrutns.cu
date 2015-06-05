/**
 * CUDA LineSearch Kernels (./cuda_sg/src/lib/lineSearch.cu)
 * Main system kernels for linesearch
 */

namespace gpu {


   /**
    * Line Discretize
    * @param N     Problem Size
    * @param D     Discrtization
    * @param x     X Vector
    * @param p     Direction Vector
    * @param h     Step Size
    * @param space Memory Space
    */
   __global__ void spcl( long int N , long int D ,  double* x, double* p, double h, double* space) {
      int i = blockDim.x * blockIdx.x + threadIdx.x;
      int j = blockDim.y * blockIdx.y + threadIdx.y;

      if (i < N && j < D) {
         space[j * N + i] = x[i] + p[i] * h * j;
      }
   };

   __global__ void spcc( long int N , double* x,  double* space) {
      int i = blockDim.x * blockIdx.x + threadIdx.x;
      int j = blockDim.y * blockIdx.y + threadIdx.y;
      space[j * N + i] = x[i];
   };


   /**
    * Function Value
    * @param N        Porbelm Size
    * @param D        Discretixation
    * @param space    Memory Space
    * @param func_val Function Value
    */
   __global__ void fv( long int N , long int D , double* space, double* func_val) {
      int i = blockDim.x * blockIdx.x + threadIdx.x;

      if (i < D ) {
         double val = 0.0;
         FUNCTION(N, &space[i * N], &val);
         func_val[i] = val;
      };
   };



   __global__ void grad( long int N ,  double EPS, double* space, double* grad) {
      int i = blockDim.x * blockIdx.x + threadIdx.x;
      double val = 0.0;

      if (i < N) {
         space[i * N + i] -= EPS;
         FUNCTION(N, &space[i * N], &val);

         space[i * N + i] += 2.0 * EPS;
         val = val * -1.0;

         FUNCTION(N, &space[i * N], &val);
         grad[i] = val / (2.0 * EPS);
      };
   };

   __global__ void axpby(const double* a, double* A, const double* b, double* B, double* rtrn) {
      int i = blockDim.x * blockIdx.x + threadIdx.x;

      if (i < N) {
         rtrn[i] = a * A[i]  + b * B[i];
      }
   };

   __global__ void dot(double* A , double* B, double* rtrn) {
      int i = blockDim.x * blockIdx.x + threadIdx.x;

      if (i < N) {
         rtrn[i] = A[i] * B[i];
      }
   };
};

