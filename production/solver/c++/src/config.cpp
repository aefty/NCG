long long int _GLB_ITR_ = 100;
long long int _GLB_ITR_LINE_ = 10;
double _GLB_EPS_ = 1e-6;
long long int _GLB_N_ = 1024 * 1;

using namespace std;

inline void FUNCTION(double* x , double& rtrn ) {
	for (int i = 0; i < _GLB_N_ - 1; ++i) {
		rtrn += 100 * (x[i + 1] - x[i] * x[i]) * (x[i + 1] - x[i] * x[i]) + (1 - x[i]) * (1 - x[i]);
	};
};

inline void GUESS( vector<double>& x0) {
	vector<double>rtrn (_GLB_N_, 3);
	x0 = rtrn ;
};
