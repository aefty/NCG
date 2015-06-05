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
#include "lib/cpu_sbrutns.cpp"
#include "lib/gpu_sbrutns.cu"
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


	// Racis
	double temp ;

	double    tol = _GLB_EPS_ + 1.0;
	int       itr = 0;
	double  min_i = 0;
	double  alpha = 1;
	double      h = _GLB_EPS_;
	int     range = 257;


	// Primary Variables
	vector<double> x0(_GLB_N_);
	GUESS(_GLB_N_, x0);
	double* _x0     = (double*) gpu::alloc(x0);

	vector<double> vtemp(_GLB_N_);
	double* _vtemp  = (double*) gpu::alloc(vtemp);

	vector<double> vtempl(range);
	double* _vtempl = (double*) gpu::alloc(vtempl);

	double* _x1     = (double*) gpu::alloc(_GLB_N_);
	double* _p      = (double*) gpu::alloc(_GLB_N_);
	double* _g00    = (double*) gpu::alloc(_GLB_N_);
	double* _g01    = (double*) gpu::alloc(_GLB_N_);
	double* _g1     = (double*) gpu::alloc(_GLB_N_);

	double     gg0  = 0.0;
	double     gg1  = 0.0;
	double       B  = 0.0;

	// Block Memory Allocation
	double* _gr_space = (double*) gpu::alloc(_GLB_N_ * _GLB_N_);
	double* _ld_space = (double*) gpu::alloc(range * _GLB_N_);
	double* _fv_space = (double*) gpu::alloc(range);

	// Block Schimatics
	dim3 nm_tpb (16, 16);
	dim3 nm_blocks(_GLB_N_ / nm_tpb.x + 1, 256);

	dim3 nn_tpb(128, 128);
	dim3 nn_blocks(_GLB_N_ / nn_tpb.x , _GLB_N_ / nn_tpb.y);

	dim3 ln_tpb (128);
	dim3 ln_blocks(_GLB_N_ / ln_tpb.x + 1);

	double t_lineSearch = 0.0;
	clock_t t_start = clock();

	// BEGIN NCG
	{
		gpu::alloc(x0, _x0);

		//cpu::linalg_grad(_GLB_N_, _GLB_EPS_, x0, p);
		gpu::spcc <<< nn_blocks , nn_tpb>>>   (_GLB_N_, _x0, _gr_space );
		gpu::grad <<< nn_blocks , nn_tpb>>>   (_GLB_N_, _GLB_EPS_, _gr_space , _p);

		//cpu::linalg_sdot( -1.0, p, p);
		gpu::axpby <<< ln_tpb , ln_blocks>>>  (_GLB_N_, -1.0, _p , 0.0, _p , _p);

		//cpu::linalg_dot(p, p, _vtemp);
		gpu::dot <<< ln_tpb , ln_blocks>>>    (_GLB_N_, _p, _p , _vtemp);
		{
			CUDA_ERR_CHECK(cudaDeviceSynchronize());
			gpu::unalloc(_vtemp, vtemp);
			gg0 = 0;

			for (int i = 0; i < _GLB_N_; ++i) {
				gg0 += vtemp[i];
			}
		}

		// Pointer Swap
		x1 = x0;
		temp = *x1;
		*x1 = *x0;
		*x0 = temp;

		while (tol > _GLB_EPS_ && itr < _GLB_ITR_) {

			cout << "|" << tol << endl;
			clock_t t_lineSearch_start = clock();

			/**
			* CODE BLOCK 2
			* Line Search
			*/
			{
				gpu::ld <<< ld_blocks , ld_tpb>>> (_GLB_N_, range, _x0 , _p, h , _ld_space);
				gpu::fv <<< fv_blocks , fv_tpb>>> (_GLB_N_, range, _ld_space ,  _vtempl);

				CUDA_ERR_CHECK(cudaDeviceSynchronize());
				gpu::unalloc(_vtempl, vtempl );

				for (int i = 1; i < vtempl.size(); i++) {
					if (vtempl[i] < vtempl[min_i]) {
						min_i = i;
					}
				};

				alpha = min_i * h;
			}
			// END LINE SEARCH


			/**
			* CODE BLOCK 3
			* Direction
			*/

			//cpu::linalg_add (1.0, x0, alpha, p, x1);
			gpu::axpby <<< ld_blocks , ld_tpb>>> (_GLB_N_, 1.0, _x0 , alpha, _p , _x1);

			t_lineSearch += (clock() - t_lineSearch_start) / (double) CLOCKS_PER_SEC;

			//cpu::linalg_grad(_GLB_N_, _GLB_EPS_, x1, g1);
			gpu::grad <<< ld_blocks , ld_tpb>>> (_GLB_N_, _GLB_EPS_, _x1 , _g1);

			//cpu::linalg_dot(g1, g1, gg1);
			gpu::dot <<< ld_blocks , ld_tpb>>>  (_GLB_N_, _g1, _g1 , _vtemp);
			{
				CUDA_ERR_CHECK(cudaDeviceSynchronize());
				gpu::unalloc(_vtemp, vtemp);
				gg1 = 0;

				for (int i = 0; i < _GLB_N_; ++i) {
					gg1 += vtemp[i];
				};
			}

			B = gg1 / gg0;

			//% p = -g1 + B * p;
			//cpu::linalg_add(-1.0, g1, B, p, p);
			gpu::axpby <<< ld_blocks , ld_tpb>>> (_GLB_N_, -1.0, _g1 , B, _p , _p);

			//% tol = norm(x1 - x0)
			//cpu::linalg_add(1.0, x1, -1.0, x0, vtemp);
			gpu::axpby <<< ld_blocks , ld_tpb>>> (_GLB_N_, 1.0, _x1 , -1.0, _x0 , _vtemp);

			//cpu::linalg_dot(vtemp, vtemp, tol);
			gpu::dot <<< ld_blocks , ld_tpb>>> (_GLB_N_, _vtemp, _vtemp , _vtemp);
			{
				CUDA_ERR_CHECK(cudaDeviceSynchronize());
				gpu::unalloc(_vtemp, vtemp);
				tol = 0;

				for (int i = 0; i < _GLB_N_; ++i) {
					tol += vtemp[i];
				};
			}

			tol = pow(tol , 0.5) / _GLB_N_;
			gg0 = gg1;

			x0 = x1;
			temp = *x0;
			*x0 = *x1;
			*x1 = temp;

			itr ++;
		}
	}
	//END NCG


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

	if (showX) {
		json.append("x", x1);
	}

	cout << "\n\n";
	cout << json.dump();
	cout << "\n\n";
	return 0;
}