namespace cuda {

   __device__ static float atomicMax(float* address, float val) {
      int* address_as_i = (int*) address;
      int old = *address_as_i, assumed;

      do {
         assumed = old;
         old = ::atomicCAS(address_as_i, assumed,
                           __float_as_int(::fmaxf(val, __int_as_float(assumed))));
      } while (assumed != old);

      return __int_as_float(old);
   }


   __global__ void discLine_kernel( long int N ,  double* x, double* p, double h, double* space) {
      int i = blockDim.x * blockIdx.x + threadIdx.x;
      int j = blockDim.y * blockIdx.y + threadIdx.y;

      cout << "(" << i << "," << j << ") " << endl;


      //space[j * N + i] = x[i] + p[i] * h * j;
   };

   /*å
      __global__ void lineValue( long int N , double* space, double* alpha_set) {
         int i = blockDim.x * blockIdx.x + threadIdx.x;
         double val = 0.0;
         FUNCTION(N, &space[i * N], &val);
         alpha_set[i] = (val < alpha_set[i]) ? val : alpha_set[i];
      };

   */

   inline void lineSearch_disc(long int N , double h, double rad, vector<double>& x, vector<double>& p, double* _space) {
      int TPB_OPTIMAL_2D = 16;

      dim3 GPU_TPB_2D(TPB_OPTIMAL_2D, TPB_OPTIMAL_2D);
      dim3 GPU_BLOCK_2D(N / GPU_TPB_2D.x , (unsigned int)rad / h);

      double* _x = (double*)cuda::alloc(x);
      double* _p = (double*)cuda::alloc(p);

      discLine_kernel <<<GPU_BLOCK_2D , GPU_TPB_2D>>> (N, _x , _p, h , _space);
   };

   /*
      inline void lineSearch_solve(long int N , double EPS, double* _space, vector<double>& x, vector<double>& p, double& alpha_last , double& alpha) {
         int TPB_OPTIMAL_1D = 256;
         int TPB_OPTIMAL_2D = 16;

         dim3 GPU_TPB_1D (TPB_OPTIMAL_1D);
         dim3 GPU_BLOCK_1D(_GLB_N_ / GPU_TPB_1D.x);



         vector<double> alpha_set(GPU_TPB_1D.x * GPU_BLOCK_1D.x);
         double* _alpha_set = (double*)cuda::alloc(alpha_set);



         discLine <<< GPU_BLOCK_2D , GPU_TPB_2D>>> (N, _x , _p, EPS , _space);
         lineValue  <<<GPU_BLOCK_1D , GPU_TPB_1D>>> (N,  _space, _alpha_set);

         cuda::unalloc(_alpha_set, alpha_set);
         cuda::unalloc(_x);
         cuda::unalloc(_p);

         alpha = *min_element(std::begin(alpha_set), std::end(alpha_set));
      };
      */
};

