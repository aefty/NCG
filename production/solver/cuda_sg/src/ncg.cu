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

	cuda::deviceSpecs();

	JSON json;

	vector<double> x0;
	GUESS(_GLB_N_, x0);

	vector<double> x1(_GLB_N_);
	vector<double> p(_GLB_N_);
	vector<double> vtemp(_GLB_N_);
	vector<double> g00(_GLB_N_);
	vector<double> g01(_GLB_N_);
	vector<double> Hp(_GLB_N_);
	vector<double> g1(_GLB_N_);

	int lineDisc = (long int) * 1024;
	double* _space = (double*) cuda::alloc(lineDisc);

	double gg0 = 0.0;
	double gg1 = 0.0;
	double B = 0.0;
	double stemp = 0.0;

	double tol = _GLB_EPS_ + 1.0;
	int itr = 0;

	int j;
	double alpha_last;
	double alpha;

	double t_lineSearch = 0.0;

	clock_t t_start = clock();

	std::cout << "NCG - Started " << endl;


	// BEGIN NCG
	{
		std::cout << "|";
		std::linalg_grad(_GLB_N_, _GLB_EPS_, x0, p);
		std::linalg_sdot( -1.0, p, p);

		std::linalg_dot(p, p, gg0);

		x1 = x0;

		while (tol > _GLB_EPS_ && itr < _GLB_ITR_) {

			std::cout << "|"; std::cout.flush();

			j = 0;
			alpha_last = 1.0;
			alpha = 2.0;

			clock_t t_lineSearch_start = clock();

			cuda::line_search(N , _GLB_EPS_,_space ], x1, p,alpha,);
			std::linalg_add(1.0, x1, alpha, p, x1);

			// BEING LINE SEARCH
			/*
			{
				while (j < _GLB_ITR_LINE_ && abs(alpha - alpha_last) >= _GLB_EPS_) {
					std::cout << "."; std::cout.flush();

					//%% Note : Calculate Hessian x p (Hp)
					//%% 2nd-term Taylor expansion (average -/+ expansion for better accuracy)

					//%g00 = Grad(x1-EPS*p);
					std::linalg_add(1.0, x1, -1.0 * _GLB_EPS_, p, vtemp);
					cuda::linalg_grad(_GLB_N_, _GLB_EPS_, vtemp, g00);

					//%g01 = Grad(x1+EPS*p);
					std::linalg_add(1.0, x1, _GLB_EPS_, p, vtemp);
					cuda::linalg_grad(_GLB_N_, _GLB_EPS_, vtemp, g01);

					// %Hp = (g01-g00)/(2*EPS);
					std::linalg_add(1.0, g01, -1.0, g00, vtemp);
					std::linalg_sdot(1.0 / (2.0 * _GLB_EPS_), vtemp,  Hp);

					alpha_last = alpha;

					//%alpha = -g00'*p/(p'*Hp);
					std::linalg_dot(g00, p, stemp);
					std::linalg_dot(p, Hp, alpha);
					alpha = -1.0 * stemp / alpha;

					// %x1=x1+alpha*p;
					std::linalg_add(1.0, x1, alpha, p, x1);

					j++;
				}
			}
			*/

			// END LINE SEARCH
			t_lineSearch += (clock() - t_lineSearch_start) / (double) CLOCKS_PER_SEC;

			std::linalg_grad(_GLB_N_, _GLB_EPS_, x1, g1);
			std::linalg_dot(g1, g1, gg1);
			B = gg1 / gg0;

			//% p = -g1 + B * p;
			std::linalg_add(-1.0, g1, B, p, p);

			//% tol = norm(x1 - x0)
			std::linalg_add(1.0, x1, -1.0, x0, vtemp);
			std::linalg_dot(vtemp, vtemp, tol);
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

	cuda::unalloc(_space);

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
	//json.append("x", x1);

	cout << "\n\n";
	cout << json.dump();
	cout << "\n\n";
	return 0;
}