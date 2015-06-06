/**
 * NON LINEAR CG SOLVER (./cuda_sg/src/ncg.cu)
 * Main solver file.
 */

#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <time.h>
#include <algorithm>

#include "config.cu"
#include "lib/util.cpp"
#include "lib/math.cpp"
#include "lib/lineSearch.cu"
#include "lib/json.cpp"

using namespace std;

int main(int argc, char* argv[]) {

	int showX = 0;

	if (argc > 1) { _GLB_N_ = (long int) _GLB_N_ * atof(argv[1]); }

	if (argc > 2) { showX = atoi(argv[2]); }

	gpu::deviceSpecs();

	/**
	 * CODE BLOCK 1
	 * Initilization
	 */

	JSON json;

	vector<double> x0; GUESS(_GLB_N_, x0); double* _x0 = (double*) gpu::alloc(x0);
	vector<double> x1(_GLB_N_);
	vector<double> p(_GLB_N_); double* _p = (double*) gpu::alloc(p);
	vector<double> vtemp(_GLB_N_);
	vector<double> g00(_GLB_N_);
	vector<double> g01(_GLB_N_);
	vector<double> g1(_GLB_N_);

	double gg0 = 0.0;
	double gg1 = 0.0;
	double B = 0.0;

	double tol = _GLB_EPS_ + 1.0;
	int itr = 0;
	double  min_i = 0;
	double alpha = 1;
	double h = _GLB_EPS_;

	// ~50% staturated
	int TPB_2D = 16 ;
	long int range = 16;

	int block_x = (_GLB_N_ / TPB_2D) < 1 ? 1 : (_GLB_N_ / TPB_2D) ;
	int block_y = range < 1 ? 1 : range ;

	dim3 nm_tpb (128);
	dim3 nm_blocks(_GLB_N_*range/128);

	dim3 ln_tpb (128);
	dim3 ln_blocks(_GLB_N_ / 128) ;

	vector<double> space(range * _GLB_N_, 0.0); double* _space = (double*) gpu::alloc(space);
	vector<double> func_val(range, 0.0); double* _func_val = (double*) gpu::alloc(func_val);

	double t_lineSearch = 0.0;
	clock_t t_start = clock();

	// BEGIN NCG
	{
		cpu::linalg_grad(_GLB_N_, _GLB_EPS_, x0, p);
		cpu::linalg_sdot( -1.0, p, p);

		cpu::linalg_dot(p, p, gg0);

		x1 = x0;

		while (tol > _GLB_EPS_ && itr < _GLB_ITR_) {

			cout << "| Tol :" << tol << endl;
			clock_t t_lineSearch_start = clock();

			/**
			* CODE BLOCK 2
			* Line Search
			*/
			{
				gpu::alloc(x0, _x0);
				gpu::alloc(p, _p);

				gpu::spcl <<< nm_tpb , nm_blocks>>>   (_GLB_N_, range, _x0 , _p, h , _space);
			//	gpu::fv <<<ln_tpb , ln_blocks>>> (_GLB_N_, range, _space ,  _func_val);

				CUDA_ERR_CHECK(cudaDeviceSynchronize());
				gpu::unalloc(_func_val, func_val );

				for (int i = 0; i < func_val.size(); i++) {
					if (func_val[i] < func_val[min_i]) {
						min_i = i;
					}
				}

				alpha = min_i * h;
			}
			// END LINE SEARCH
goto end;

			/**
			* CODE BLOCK 3
			* Direction
			*/
			cpu::linalg_add (1.0, x0, alpha, p, x1);

			t_lineSearch += (clock() - t_lineSearch_start) / (double) CLOCKS_PER_SEC;

			cpu::linalg_grad(_GLB_N_, _GLB_EPS_, x1, g1);
			cpu::linalg_dot(g1, g1, gg1);
			B = gg1 / gg0;

			//% p = -g1 + B * p;
			cpu::linalg_add(-1.0, g1, B, p, p);

			//% tol = norm(x1 - x0)
			cpu::linalg_add(1.0, x1, -1.0, x0, vtemp);
			cpu::linalg_dot(vtemp, vtemp, tol);
			tol = pow(tol , 0.5) / _GLB_N_;
			gg0 = gg1;

			x0 = x1;
			itr ++;
		}
	}
	//END NCG

end:
	// Get timining and metrics
	double t_run = (clock() - t_start) / (double) CLOCKS_PER_SEC;
	double rate = (double)_GLB_N_ / t_run;
	t_lineSearch = t_lineSearch;

	gpu::unalloc(_space, space);
	gpu::unalloc(_space);

	double x_max = *max_element(std::begin(x1), std::end(x1));
	double x_min = *min_element(std::begin(x1), std::end(x1));

	// Output
	json.append("size", _GLB_N_);
	json.append("itr", itr);
	json.append("conv", tol);
	json.append("run_time", t_run);
	json.append("line_search_time", t_lineSearch);
	json.append("rate", rate);
	json.append("x_max", x_max);
	json.append("x_min", x_min);
	
	json.append("space", space);
	json.append("func_val", func_val);

	if (showX) {
		json.append("x", x1);
	}

	cout << "\n\n";
	cout << json.dump();
	cout << "\n\n";
	return 0;
}