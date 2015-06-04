#include <vector>

#define CUDA_ERR_CHECK(x) \
	do { cudaError_t err = x; if (err != cudaSuccess) { \
			fprintf (stderr, "Error \"%s\" at %s:%d \n", \
			         cudaGetErrorString(err), \
			         __FILE__, __LINE__); exit(-1); \
		}} while (0);


namespace cuda {

	inline void* alloc(long long int N) {
		void* p;
		size_t s = N * sizeof(double);
		CUDA_ERR_CHECK(cudaMalloc(&p, s));
		return p;
	};

	inline void* alloc(vector<double>& host_vec) {
		void* p;
		size_t s = host_vec.size() * sizeof(double);
		double* host_array = &host_vec[0];
		CUDA_ERR_CHECK(cudaMalloc(&p, s));
		CUDA_ERR_CHECK(cudaMemcpy(p, host_array, s, cudaMemcpyHostToDevice));
		return p;
	};

	inline void* alloc(double& host_scal) {
		void* p;
		size_t s = sizeof(double);
		CUDA_ERR_CHECK(cudaMalloc(&p, s));
		CUDA_ERR_CHECK(cudaMemcpy(p, &host_scal, s, cudaMemcpyHostToDevice));
		return p;
	};

	inline void unalloc(double* p , vector<double>& host_vec) {
		size_t s = host_vec.size() * sizeof(double);
		CUDA_ERR_CHECK(cudaMemcpy(&host_vec[0], p, s, cudaMemcpyDeviceToHost));
		cudaFree(p);
	};

	inline void unalloc(double* p , double& host_scal) {
		size_t s = sizeof(double);
		CUDA_ERR_CHECK(cudaMemcpy(&host_scal, p, s, cudaMemcpyDeviceToHost));
		cudaFree(p);
	};

	inline void unalloc(double* p) {
		cudaFree(p);
	};
};