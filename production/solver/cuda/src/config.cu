/**
 * SOLVER CONFIGUATION (./cuda/src/config.cpp)
 * Global configuration of the solver including the function and intial value are specfied.
 */
#include <math.h>       /* exp */
long int _GLB_N_ = 1024 * 1;  // Probelm Size - Note keep at base 2
long int _GLB_ITR_ = 10;     // Max Solver Iterations
long int _GLB_ITR_LINE_ = 2; // Max Line search iterationsc
double _GLB_EPS_ = 1e-6;      // Value of epsilon, Note this is equal to the tolerence (both residual and linesearch)


using namespace std;

__device__ void _FUNCTION(long int N, double* x , double* rtrn ) {
	for (int i = 0; i <  N - 1; ++i) {
		rtrn[0] += 100 * (x[i + 1] - x[i] * x[i]) * (x[i + 1] - x[i] * x[i]) + (1 - x[i]) * (1 - x[i]);
	};
};

__device__ void FUNCTION(long int N, double* x , double* rtrn ) {
	for (int i = 0; i <  N ; ++i) {
		rtrn[0] += (x[i]-1.0) / (exp ((x[i]-1.0)/.1)-1.0);
	};
};

//\frac{\left(x-1\right)}{e^{\frac{\left(x-1\right)}{.1}}-1}

inline void GUESS(long int  N, vector<double>& rtrn) {
	vector<double>x0 (N, -7);
	rtrn = x0 ;
};
