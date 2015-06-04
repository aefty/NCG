namespace cuda {

   //
   __global__ void initSpace( long int N ,  double* x, double EPS, double* space) {
      int i = blockDim.x * blockIdx.x + threadIdx.x;
      int j = blockDim.y * blockIdx.y + threadIdx.y;
      space[j * N + i] = x[i] + EPS;
   };

   __global__ void async_grad( long int N , int i,  double EPS, double* x, double* grad) {
      double val = 0.0;

      x[i] -= EPS;
      FUNCTION(N, &x[0], &val);

      x[i] += 2.0 * EPS;
      val = val * -1.0;

      FUNCTION(N, &x[0], &val);
      grad[0] = val / (2.0 * EPS);
   };

   inline void linalg_grad(long int N ,  double EPS, vector<double>& x,  vector<double>& grad) {

      // Create streams
      cudaStream_t stream[N];
      int streamLength = 1;
      int streamSize = streamLength * sizeof(double);


      for (int i = 0; i < N; i++) {

         double* _x = (double*)cuda::alloc(x);
         double* _gradi = (double*)cuda::alloc(grad[i]);

         CUDA_ERR_CHECK(cudaStreamCreate(&stream[i]));
         CUDA_ERR_CHECK(cudaMemcpyAsync( _x, &x[0], streamSize, cudaMemcpyHostToDevice, stream[i])) ;

         async_grad <<<1, 1, 0, stream[i]>>> (N, i, EPS, _x, _gradi);

         CUDA_ERR_CHECK(cudaMemcpyAsync( _gradi, &grad[i], streamSize, cudaMemcpyDeviceToHost, stream[i])) ;
      }

      cudaThreadSynchronize();
   };
};

