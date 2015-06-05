/**
 * CUDA HELPER FUNCTIONS (./cuda_sg/src/lib/util.cpp)
 * Help functions mainly for cuda memory allocation.
 */


#include <vector>

/**
 * Cuda Error Check Macro (Taken from HPC Asignment - USI 2015)
 * @param  x Input
 * @return   Text Error output
 */
#define CUDA_ERR_CHECK(x) \
	do { cudaError_t err = x; if (err != cudaSuccess) { \
			fprintf (stderr, "Error \"%s\" at %s:%d \n", \
			         cudaGetErrorString(err), \
			         __FILE__, __LINE__); exit(-1); \
		}} while (0);


namespace gpu {

	/**
	 * Allocate Memory - Copy input vector to device pointer, return pointer (consitency)
	 */
	inline void* alloc(vector<double>& host_vec, void* p) {
		size_t s = host_vec.size() * sizeof(double);
		double* host_array = &host_vec[0];
		CUDA_ERR_CHECK(cudaMemcpy(p, host_array, s, cudaMemcpyHostToDevice));
		return p;
	};

	/**
	 * Allocate Memory -  New memory block, return device pointer
	 */
	inline void* alloc(long long int N) {
		void* p;
		size_t s = N * sizeof(double);
		CUDA_ERR_CHECK(cudaMalloc(&p, s));
		return p;
	};

	/**
	 * Allocate Memory - Copy input vector, return device pointer
	 */
	inline void* alloc(vector<double>& host_vec) {
		void* p;
		size_t s = host_vec.size() * sizeof(double);
		double* host_array = &host_vec[0];
		CUDA_ERR_CHECK(cudaMalloc(&p, s));
		CUDA_ERR_CHECK(cudaMemcpy(p, host_array, s, cudaMemcpyHostToDevice));
		return p;
	};

	/**
	 * Allocate Memory -  Copy input scalar, return device pointer
	 */
	inline void* alloc(double& host_scal) {
		void* p;
		size_t s = sizeof(double);
		CUDA_ERR_CHECK(cudaMalloc(&p, s));
		CUDA_ERR_CHECK(cudaMemcpy(p, &host_scal, s, cudaMemcpyHostToDevice));
		return p;
	};

	/**
	 * Unallocate Memory -  Copy device pointer into host vector (size from host vector)
	 */
	inline void unalloc(double* p , vector<double>& host_vec) {
		size_t s = host_vec.size() * sizeof(double);
		CUDA_ERR_CHECK(cudaMemcpy(&host_vec[0], p, s, cudaMemcpyDeviceToHost));
	};

	/**
	 * Unallocate Memory - Copy device scalar into host scaalar
	 */
	inline void unalloc(double* p , double& host_scal) {
		size_t s = sizeof(double);
		CUDA_ERR_CHECK(cudaMemcpy(&host_scal, p, s, cudaMemcpyDeviceToHost));
	};

	/**
	 * Unallocate Memory - Delete device memory.
	 */
	inline void unalloc(double* p) {
		cudaFree(p);
	};

	/**
	 * CUDA Device Spec Tempalte (Source Nvida.com)
	 */
	inline void deviceSpecs() {
		const int kb = 1024;
		const int mb = kb * kb;

		std::cout << "NCG - Started " << endl;
		std::cout << "CUDA version:   v" << CUDART_VERSION << endl;

		int devCount;
		cudaGetDeviceCount(&devCount);
		std::cout << "CUDA Devices: " << endl << endl;

		for (int i = 0; i < devCount; ++i) {
			cudaDeviceProp props;
			cudaGetDeviceProperties(&props, i);
			std::cout << i << ": " << props.name << ": " << props.major << "." << props.minor << endl;
			std::cout << "  Global memory:   " << props.totalGlobalMem / mb << "mb" << endl;
			std::cout << "  Shared memory:   " << props.sharedMemPerBlock / kb << "kb" << endl;
			std::cout << "  Constant memory: " << props.totalConstMem / kb << "kb" << endl;
			std::cout << "  Block registers: " << props.regsPerBlock << endl << endl;

			std::cout << "  Warp size:         " << props.warpSize << endl;
			std::cout << "  Threads per block: " << props.maxThreadsPerBlock << endl;
			std::cout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1]  << ", " << props.maxThreadsDim[2] << " ]" << endl;
			std::cout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1]  << ", " << props.maxGridSize[2] << " ]" << endl;
			std::cout << endl;
		}
	};
};