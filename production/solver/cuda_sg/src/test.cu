#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <algorithm>

#include "config.cu"
#include "lib/util.cpp"
#include "lib/math.cpp"
#include "lib/math.cu"
#include "lib/json.cpp"

using namespace std;

__global__ void discLine_kernel( long int N ,  double* x, double* p, double h, double* space) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	printf("%d , %d\n", i, j);


	//space[j * N + i] = x[i] + p[i] * h * j;
};


int main(int argc, char* argv[]) {

	if (argc > 1) { _GLB_N_ = _GLB_N_ * atoi(argv[1]); }

	if (argc > 2) { _GLB_ITR_ = _GLB_ITR_ * atoi(argv[2]); }

	if (argc > 3) { _GLB_ITR_LINE_ = _GLB_ITR_LINE_ * atoi(argv[3]); }

	if (argc > 4) { _GLB_EPS_ = _GLB_EPS_ * atoi(argv[4]); }

	cuda::deviceSpecs();

	JSON json;

	vector<double> A(_GLB_N_, 1.0);
	vector<double> B(_GLB_N_, 1.0);
	vector<double> C(_GLB_N_, 1.0);
	vector<double> D(_GLB_N_, 1.0);

	vector<double> GRAD(_GLB_N_, 0.0);


	double h = 0.001;
	double disc = 4;

	vector<double> space(disc, 0.0);
	double* _space = (double*) cuda::alloc(space);

	double scalar = 1.0;

	clock_t t_start_dot = clock();
	std::linalg_dot (A, B, scalar);
	double t_dot = (clock() - t_start_dot) / (double) CLOCKS_PER_SEC;

	clock_t t_start_sdot = clock();
	std::linalg_sdot(scalar, A, B);
	double t_sdot = (clock() - t_start_sdot) / (double) CLOCKS_PER_SEC;

	clock_t t_start_add = clock();
	std::linalg_add (1.0, A, 1.0, B, C);
	double t_add = (clock() - t_start_add) / (double) CLOCKS_PER_SEC;

	clock_t t_start_grad_cuda = clock();

	{

		int TPB_OPTIMAL_1D = 128;
		int blocks = (_GLB_N_ * disc / TPB_OPTIMAL_1D) < 1 ? 1 : _GLB_N_ * disc / TPB_OPTIMAL_1D  ;

		cout << TPB_OPTIMAL_1D << endl;
		cout << blocks << endl;

		double* _x = (double*)cuda::alloc(A);
		double* _p = (double*)cuda::alloc(A);

		discLine_kernel <<< blocks , TPB_OPTIMAL_1D>>> (_GLB_N_, _x , _p, h , _space);
	}


	//cuda::lineSearch_disc(_GLB_N_ , h, disc,  A, A, _space);


	double t_grad_cuda = (clock() - t_start_grad_cuda) / (double) CLOCKS_PER_SEC;

	double max_grad = *max_element(std::begin(C), std::end(C));
	double min_grad = *min_element(std::begin(C), std::end(C));


	cudaDeviceSynchronize();

	cuda::unalloc(_space, space );
	cuda::unalloc(_space);

	json.append("size", _GLB_N_);
	json.append("dot_time", t_dot);
	json.append("sdot_time", t_sdot);
	json.append("add_time", t_add);
	json.append("cuda_grad_time", t_grad_cuda);
	json.append("min", min_grad);
	json.append("max", max_grad);
	json.append("space", space);
	//json.append("C", C);

	cout << "\n\n" << json.dump() << "\n\n";
	return 0;
}
