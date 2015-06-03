<<<<<<< HEAD
/**
 * SOLVER CONFIGUATION (./cuda/src/config.cpp)
 * Global configuration of the solver including the function and intial value are specfied.
 */

const long int _GLB_ITR_ = 100;     // Max Solver Iterations
const long int _GLB_ITR_LINE_ = 10; // Max Line search iterationsc
const double _GLB_EPS_ = 1e-6;      // Value of epsilon, Note this is equal to the tolerence (both residual and linesearch)
const long int _GLB_N_ = 1024 * 1;  // Probelm Size - Note keep at base 2

using namespace std;

__device__ void FUNCTION(const long int N, double* x , double* rtrn ) {
=======
long long int _GLB_ITR_ = 10000;
long long int _GLB_ITR_LINE_ = 100;
double _GLB_EPS_ = 1e-6;
long long int _GLB_N_ = 1024*1;

using namespace std;

__device__ void FUNCTION(long int N, double* x , double* rtrn ) {
>>>>>>> 2121b1afb91f1e00bbd53ed5101ab1041e79a4f6
	for (int i = 0; i <  N - 1; ++i) {
		rtrn[0] += 100 * (x[i + 1] - x[i] * x[i]) * (x[i + 1] - x[i] * x[i]) + (1 - x[i]) * (1 - x[i]);
	};
};

<<<<<<< HEAD
inline void GUESS(const long int  N, vector<double>& x0) {
=======
inline void GUESS( long int  N, vector<double>& x0) {
>>>>>>> 2121b1afb91f1e00bbd53ed5101ab1041e79a4f6
	vector<double>rtrn (N, 7);
	x0 = rtrn ;
};
