/**
 * SOLVER CONFIGUATION (./cuda/src/config.cpp)
 * Global configuration of the solver including the function and intial value are specfied.
 */

const long int _GLB_ITR_ = 100;     // Max Solver Iterations
const long int _GLB_ITR_LINE_ = 10; // Max Line search iterationsc
const double _GLB_EPS_ = 1e-6;      // Value of epsilon, Note this is equal to the tolerence (both residual and linesearch)
const long int _GLB_N_ = 1024 * 20;  // Probelm Size - Note keep at base 2

using namespace std;

__device__ void FUNCTION(const long int N, double* x , double* rtrn ) {
	for (int i = 0; i <  N - 1; ++i) {
		rtrn[0] += 100 * (x[i + 1] - x[i] * x[i]) * (x[i + 1] - x[i] * x[i]) + (1 - x[i]) * (1 - x[i]);
	};
};

inline void GUESS(const long int  N, vector<double>& rtrn) {
	vector<double>x0 (N, 7);
	rtrn = x0 ;
};
