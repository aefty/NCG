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
#include "lib/math.cu"
#include "lib/math.cpp"
#include "lib/json.cpp"

using namespace std;

int main(int argc, char* argv[]) {

	if (argc > 1) { _GLB_N_ = (long int) _GLB_N_ * atof(argv[1]); }

	if (argc > 2) { _GLB_ITR_ = (long int)_GLB_ITR_ * atof(argv[2]); }

	if (argc > 3) { _GLB_ITR_LINE_ = (long int)_GLB_ITR_LINE_ * atof(argv[3]); }

	if (argc > 4) { _GLB_EPS_ = (double) _GLB_EPS_ * atof(argv[4]); }

	cout << _GLB_ITR_ << endl;

	gpu::deviceSpecs();

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
	int  min_i = 0;
	double alpha = 1;
	double h = 1;


	int TPB_2D = 16 ;
	long int range = 16;

	int block_x = (_GLB_N_ / TPB_2D) < 1 ? 1 : (_GLB_N_ / TPB_2D) ;
	int block_y = range < 1 ? 1 : range ;

	dim3 GPU_TPB_2D (TPB_2D, TPB_2D);
	dim3 GPU_BLOCK_2D(block_x , block_y);

	dim3 GPU_TPB_1D (TPB_2D * TPB_2D);
	dim3 GPU_BLOCK_1D(_GLB_N_ / GPU_TPB_1D.x + 1) ;

	vector<double> space(range * _GLB_N_, 0.0); double* _space = (double*) gpu::alloc(space);
	vector<double> func_val(range, 0.0); double* _func_val = (double*) gpu::alloc(func_val);



	double t_lineSearch = 0.0;


	clock_t t_start = clock();

	std::cout << "NCG - Started " << endl;


	// BEGIN NCG
	{
		std::cout << "|";

		cpu::linalg_grad(_GLB_N_, _GLB_EPS_, x0, p);
		cpu::linalg_sdot( -1.0, p, p);

		cpu::linalg_dot(p, p, gg0);

		x1 = x0;

		while (tol > _GLB_EPS_ && itr < _GLB_ITR_) {

			std::cout << "|"; std::cout.flush();

			clock_t t_lineSearch_start = clock();

			// BEGIN LINE SEARCH

			gpu::alloc(x0, _x0);
			gpu::alloc(p, _p);

		redo:
			h  = h * 2;
			gpu::lineDiscretize <<< GPU_BLOCK_2D , GPU_TPB_2D>>>   (_GLB_N_, range, _x0 , _p, h , _space);
			gpu::lineValue <<<GPU_BLOCK_1D , GPU_TPB_1D>>> (_GLB_N_, range, _space ,  _func_val);

			CUDA_ERR_CHECK(cudaDeviceSynchronize());
			gpu::unalloc(_func_val, func_val );

			for (int i = 1; i < func_val.size(); i++) {
				if (func_val[i] < func_val[min_i]) {
					min_i = i;
				}
			}

			alpha = min_i * h;

			if (alpha == 0 && h > _GLB_EPS_) {
				h = h / 2;
				std::cout << "."; std::cout.flush();
				goto redo;
			}

			cpu::linalg_add (1.0, x0, alpha, p, x1);


			// END LINE SEARCH

			t_lineSearch += (clock() - t_lineSearch_start) / (double) CLOCKS_PER_SEC;

			cpu::linalg_grad(_GLB_N_, _GLB_EPS_, x1, g1);
			cpu::linalg_dot(g1, g1, gg1);
			B = gg1 / gg0;

			//% p = -g1 + B * p;
			cpu::linalg_add(-1.0, g1, B, p, p);

			//% tol = norm(x1 - x0)
			cpu::linalg_add(1.0, x1, -1.0, x0, vtemp);
			cpu::linalg_dot(vtemp, vtemp, tol);
			tol = pow(tol , 0.5);
			gg0 = gg1;
			x0 = x1;

			itr ++;
		}
	}
	//END NCG

	double t_run = (clock() - t_start) / (double) CLOCKS_PER_SEC;
	double rate = (double)_GLB_N_ / t_run;
	t_lineSearch = t_lineSearch;

	gpu::unalloc(_space, space);
	gpu::unalloc(_space);

	double x_max = *max_element(std::begin(x1), std::end(x1));
	double x_min = *min_element(std::begin(x1), std::end(x1));

	json.append("size", _GLB_N_);
	json.append("itr", itr);
	json.append("conv", tol);
	json.append("run_time", t_run);
	json.append("line_search_time", t_lineSearch);
	json.append("rate", rate);
	json.append("x_max", x_max);
	json.append("x_min", x_min);
	//json.append("func_val", func_val);
	//json.append("p", p);
	//json.append("x0", x0);
	json.append("x1", x1);
	//json.append("min_i", min_i);
	//json.append("alpha", alpha);
	//json.append("space", space);

	cout << "\n\n";
	cout << json.dump();
	cout << "\n\n";
	return 0;
}