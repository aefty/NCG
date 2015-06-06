/**
 * SOLVER CONFIGUATION (./cuda_sg/src/config.cpp)
 * Global configuration of the solver including the function and intial value are specfied.
 */

long int _GLB_N_ = 4 * 1;  // Probelm Size - Note keep at base 2
long int _GLB_ITR_ = 1;     // Max Solver Iterations
long int _GLB_ITR_LINE_ = 0;  // Not relevant here
double _GLB_EPS_ = 1e-6;      // Value of epsilon, Note this is equal to the tolerence (both residual and linesearch)

using namespace std;

__host__ __device__ void __FUNCTION(long int N, double* x , double* rtrn ) {
	for (int i = 0; i <  N - 1; ++i) {
		rtrn[0] += 100 * (x[i + 1] - x[i] * x[i]) * (x[i + 1] - x[i] * x[i]) + (1 - x[i]) * (1 - x[i]);
	};
};


__host__ __device__ void FUNCTION(long int N, double* x , double* rtrn ) {
	for (int i = 0; i <  N; ++i) {
		rtrn[0] += x[i]*x[i];
	};
};

__host__ __device__ void _FUNCTION(long int N, double* x , double* rtrn ) {
	for (int i = 0; i <  N; ++i) {
		rtrn[0] += x[i];
	};
};

inline void GUESS(long int  N, vector<double>& rtrn) {
	vector<double>x0 (N, 7);
	rtrn = x0 ;
};
