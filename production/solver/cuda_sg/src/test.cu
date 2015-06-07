#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <algorithm>

#include "config.cu"
#include "lib/util.cpp"
#include "lib/cpu_subr.cpp"
#include "lib/gpu_subr.cu"
#include "lib/json.cpp"

using namespace std;


int main(int argc, char* argv[]) {

	int showX = 0;
	int showTol = 0;

	if (argc > 1) { _GLB_N_ = (long int) _GLB_N_ * atof(argv[1]); }

	if (argc > 2) { showX = atoi(argv[2]); }

	if (argc > 3) { showTol = atoi(argv[3]); }

	gpu::deviceSpecs();

	JSON json;

	vector<double> A(_GLB_N_, 2.0); double* _A = (double*)gpu::alloc(A);
	vector<double> B(_GLB_N_, 1.0);
	vector<double> C(_GLB_N_, 1.0);

	vector<double> P(_GLB_N_); double* _P = (double*)gpu::alloc(P);

	// Line Descretiztion
	int TPB_2D = 16 ;
	long int range = 16;

	int rows = (_GLB_N_ / TPB_2D) < 1 ? 1 : (_GLB_N_ / TPB_2D) ;
	int cols = range < 1 ? 1 : range ;

	dim3 GPU_TPB_2D (TPB_2D, TPB_2D);
	dim3 GPU_BLOCK_2D(rows , cols);

	vector<double> space(range * _GLB_N_, 0.0); double* _space = (double*) gpu::alloc(space);
	vector<double> func_val(range, 0.0); double* _func_val = (double*) gpu::alloc(func_val);

	double scalar = 1.0;


	clock_t t_start_dot = clock();
	cpu::linalg_dot (A, B, scalar);
	double t_dot = (clock() - t_start_dot) / (double) CLOCKS_PER_SEC;

	clock_t t_start_sdot = clock();
	cpu::linalg_sdot(scalar, A, B);
	double t_sdot = (clock() - t_start_sdot) / (double) CLOCKS_PER_SEC;

	clock_t t_start_add = clock();
	cpu::linalg_add (1.0, A, 1.0, B, C);
	double t_add = (clock() - t_start_add) / (double) CLOCKS_PER_SEC;


	int  min_i = 0;

	clock_t t_start_lineSearch = clock();
	double h = .5;
	gpu::lineDiscretize <<<GPU_BLOCK_2D , GPU_TPB_2D>>>   (_GLB_N_, range, _A , _P, h , _space);
	gpu::lineValue <<< (_GLB_N_ / 128 + 1), 128 >>> (_GLB_N_, range, _space ,  _func_val);
	gpu::unalloc(_func_val, func_val );

	for (int i = 1; i < func_val.size(); i++) {
		if (func_val[i] < func_val[min_i]) {
			min_i = i;
		}
	}

	cpu::linalg_add (1.0, A, min_i * h, P, A);
	double t_lineSearch = (clock() - t_start_lineSearch) / (double) CLOCKS_PER_SEC;


	double max_grad = *max_element(std::begin(A), std::end(A));
	double min_grad = *min_element(std::begin(A), std::end(A));


	cudaDeviceSynchronize();

	gpu::unalloc(_space, space );
	gpu::unalloc(_space);
	gpu::unalloc(_func_val);
	gpu::unalloc(_A);
	gpu::unalloc(_P);

	json.append("size", _GLB_N_);
	json.append("dot_time", t_dot);
	json.append("sdot_time", t_sdot);
	json.append("add_time", t_add);
	json.append("lineSearch_time", t_lineSearch);
	json.append("min", min_grad);
	json.append("max", max_grad);
	//json.append("space", space);
	//json.append("func_val", func_val);
	//json.append("A_min", A);
	//json.append("C", C);

	cout << "\n\n" << json.dump() << "\n\n";
	return 0;
}
