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


      const int blockSize = 1, nStreams = N;
      const int n = N;
      const int streamSize = n / nStreams;
      const int streamBytes = streamSize * sizeof(double);
      const int bytes = n * sizeof(double);

      // allocate pinned host memory and device memory
      double* x, *_x;
      CUDA_ERR_CHECK( cudaMallocHost((void**)&x, bytes) );      // host pinned
      CUDA_ERR_CHECK( cudaMalloc((void**)&_x, bytes) ); // device

      // create events and streams
      cudaEvent_t startEvent, stopEvent, dummyEvent;
      cudaStream_t stream[nStreams];
      CUDA_ERR_CHECK( cudaEventCreate(&startEvent) );
      CUDA_ERR_CHECK( cudaEventCreate(&stopEvent) );
      CUDA_ERR_CHECK( cudaEventCreate(&dummyEvent) );


      // asynchronous version 1: loop over {copy, kernel, copy}
      //memset(a, 0, bytes);
      CUDA_ERR_CHECK( cudaEventRecord(startEvent, 0) );

      for (int i = 0; i < nStreams; ++i) {

         CUDA_ERR_CHECK( cudaStreamCreate(&stream[i]) );



         int offset = i * streamSize;
         CUDA_ERR_CHECK( cudaMemcpyAsync(&d_a[offset], &a[offset],
                                         streamBytes, cudaMemcpyHostToDevice,
                                         stream[i]) );
         kernel <<< streamSize / blockSize, blockSize, 0, stream[i] >>> (d_a, offset);
         CUDA_ERR_CHECK( cudaMemcpyAsync(&a[offset], &d_a[offset],
                                         streamBytes, cudaMemcpyDeviceToHost,
                                         stream[i]) );
      }

      CUDA_ERR_CHECK( cudaEventRecord(stopEvent, 0) );
      CUDA_ERR_CHECK( cudaEventSynchronize(stopEvent) );

      // cleanup
      checkCuda( cudaEventDestroy(startEvent) );
      checkCuda( cudaEventDestroy(stopEvent) );
      checkCuda( cudaEventDestroy(dummyEvent) );

      for (int i = 0; i < nStreams; ++i) {
         checkCuda( cudaStreamDestroy(stream[i]) );
      }

      cudaFree(_x);
      cudaFreeHost(x);

   };
   /*
   inline void linalg_grad(long int N ,  double EPS, vector<double>& x,  vector<double>& grad) {

      // Create streams
      cudaStream_t stream[N];
      int streamLength = 1;
      int streamSize = streamLength * sizeof(double);
      size_t s = x.size() * sizeof(double);

      for (int i = 0; i < N; i++) {

         double* _x;
         CUDA_ERR_CHECK(cudaMalloc(_x, s));

         double* _gradi;
         CUDA_ERR_CHECK(cudaMalloc(_gradi, sizeof(double)));

         cudaEvent_t startEvent, stopEvent;

         CUDA_ERR_CHECK(cudaStreamCreate(&stream[i]));
         //CUDA_ERR_CHECK( cudaEventRecord(startEvent, stream[i]) );

         CUDA_ERR_CHECK(cudaMemcpyAsync( _x, &x[0], streamSize, cudaMemcpyHostToDevice, stream[i])) ;

         async_grad <<< 1, 1, 0, stream[i]>>> (N, i, EPS, _x, _gradi);

         CUDA_ERR_CHECK(cudaMemcpyAsync( &grad[i], _gradi,  streamSize, cudaMemcpyDeviceToHost, stream[i])) ;

       //  CUDA_ERR_CHECK( cudaEventRecord(stopEvent, stream[i]) );
       //  CUDA_ERR_CHECK( cudaEventSynchronize(stopEvent) )
         cudaFree(_x);
         cudaFree(_gradi);
      }
   };
   */
};

