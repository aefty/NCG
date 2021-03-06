/**
 * TEST FILE (./c++/src/test.cpp)
 * Operation performance test file.
 */

#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <algorithm>

#include "config.cpp"
#include "lib/cpu_subr.cpp"
#include "lib/json.cpp"

using namespace std;

int main(int argc, char* argv[]) {

	int showX = 0;
	int showTol = 0;

	if (argc > 1) { _GLB_N_ = (long int) _GLB_N_ * atof(argv[1]); }

	if (argc > 2) { showX = atoi(argv[2]); }

	if (argc > 3) { showTol = atoi(argv[3]); }

	JSON json;

	vector<double> A(_GLB_N_, 1.0);
	vector<double> B(_GLB_N_, 1.0);
	vector<double> C(_GLB_N_, 1.0);
	vector<double> D(_GLB_N_, 1.0);

	vector<double> x(_GLB_N_, 1);
	vector<double> GRAD(_GLB_N_, 0.0);

	double scalar = 1.0;

	clock_t t_start_dot = clock();
	cpu::linalg_dot (A, B, scalar);
	double t_dot = (clock() - t_start_dot) / (double) CLOCKS_PER_SEC;

	clock_t t_start_sdot = clock();
	cpu::linalg_sdot(scalar, A, C);
	double t_sdot = (clock() - t_start_sdot) / (double) CLOCKS_PER_SEC;

	clock_t t_start_add = clock();
	cpu::linalg_add (1.0, B, 1.0, C, A);
	double t_add = (clock() - t_start_add) / (double) CLOCKS_PER_SEC;

	clock_t t_start_grad = clock();
	cpu::linalg_grad(_GLB_N_, _GLB_EPS_, C,  A);
	double t_grad = (clock() - t_start_grad) / (double) CLOCKS_PER_SEC;

	clock_t t_start_grad_noWriteBack = clock();
	cpu::linalg_grad_noWriteBack(_GLB_N_, _GLB_EPS_, C);
	double t_grad_noWriteBack = (clock() - t_start_grad_noWriteBack) / (double) CLOCKS_PER_SEC;

	json.append("size", _GLB_N_);
	json.append("dot_time", t_dot);
	json.append("sdot_time", t_sdot);
	json.append("add_time", t_add);
	json.append("grad_time", t_grad);
	json.append("grad_noWriteBack", t_grad_noWriteBack);

	cout << "\n\n" << json.dump() << "\n\n";
	return 0;
}
