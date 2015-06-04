namespace cuda {

   //
   __global__ void initSpace( long int N ,  double* x, double EPS, double* space) {
      int i = blockDim.x * blockIdx.x + threadIdx.x;
      int j = blockDim.y * blockIdx.y + threadIdx.y;
      space[j * N + i] = x[i];
   };

   __global__ void bulkGrad( long int N ,  double EPS, double* space, double* grad) {
      int i = blockDim.x * blockIdx.x + threadIdx.x;
      grad[i] = 0.0;

      if (i < N) {
         space[i * N + i] -= EPS;
         FUNCTION(N, &space[i * N], &grad[i]);

         space[i * N + i] += 2.0 * EPS;
         grad[i] = grad[i] * -1.0;

         FUNCTION(N, &space[i * N], &grad[i]);
         grad[i] = grad[i] / (2.0 * EPS);
      }
   };

   __global__ void bulkGrad_test( long int N ,  double EPS, double* space, double* grad) {
      int i = blockDim.x * blockIdx.x + threadIdx.x;
      //grad[i] = 0.0;

      if (i < N) {
         // space[i * N + i] -= EPS;
         FUNCTION(N, &space[i * N], &grad[i]);

         // space[i * N + i] += 2.0 * EPS;
         // grad[i] = grad[i] * -1.0;

         FUNCTION(N, &space[i * N], &grad[i]);
         // grad[i] = grad[i] / (2.0 * EPS);
      }
   };

   inline void linalg_grad( long int N ,  double EPS, vector<double>& x,  vector<double>& grad, double* _space) {

      int TPB_OPTIMAL_1D = 256;
      int TPB_OPTIMAL_2D = 16;

      double* _grad = (double*)cuda::alloc(grad);
      double* _x = (double*)cuda::alloc(x);

      dim3 GPU_TPB_1D (TPB_OPTIMAL_1D);
      dim3 GPU_BLOCK_1D(_GLB_N_ / GPU_TPB_1D.x);

      dim3 GPU_TPB_2D(TPB_OPTIMAL_2D, TPB_OPTIMAL_2D);
      dim3 GPU_BLOCK_2D(_GLB_N_ / GPU_TPB_2D.x , _GLB_N_ / GPU_TPB_2D.y);

      initSpace <<< GPU_BLOCK_2D , GPU_TPB_2D>>> (N, _x , EPS , _space);
      bulkGrad  <<< GPU_BLOCK_1D , GPU_TPB_1D>>> (N, EPS, _space, _grad);

      cuda::unalloc(_grad, grad);
      cuda::unalloc(_x);
   };

   inline void linalg_grad_spaceAlloc( long int N ,  double EPS, vector<double>& x,  vector<double>& grad, double* _space) {

      int TPB_OPTIMAL_1D = 256;
      int TPB_OPTIMAL_2D = 16;

      double* _grad = (double*)cuda::alloc(grad);
      double* _x = (double*)cuda::alloc(x);

      dim3 GPU_TPB_1D (TPB_OPTIMAL_1D);
      dim3 GPU_BLOCK_1D(_GLB_N_ / GPU_TPB_1D.x);

      dim3 GPU_TPB_2D(TPB_OPTIMAL_2D, TPB_OPTIMAL_2D);
      dim3 GPU_BLOCK_2D(_GLB_N_ / GPU_TPB_2D.x , _GLB_N_ / GPU_TPB_2D.y);

      // initSpace <<< GPU_BLOCK_2D , GPU_TPB_2D>>> (N, _x , EPS , _space);
      bulkGrad_test  <<<GPU_BLOCK_1D , GPU_TPB_1D>>> (N, EPS, _space, _grad);

      cuda::unalloc(_grad, grad);
      cuda::unalloc(_x);
   };
};

