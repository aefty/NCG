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
#include "lib/math.cpp"
#include "lib/json.cpp"

using namespace std;

int main(int argc, char* argv[]) {

	if (argc > 1) { _GLB_N_ = _GLB_N_ * atoi(argv[1]); }

	if (argc > 2) { _GLB_ITR_ = _GLB_ITR_ * atoi(argv[2]); }

	if (argc > 3) { _GLB_ITR_LINE_ = _GLB_ITR_LINE_ * atoi(argv[3]); }

	if (argc > 4) { _GLB_EPS_ = _GLB_EPS_ * atoi(argv[4]); }

	JSON json;

	vector<double> A(_GLB_N_, 1.0);
	vector<double> B(_GLB_N_, 1.0);
	vector<double> C(_GLB_N_, 1.0);
	vector<double> D(_GLB_N_, 1.0);

	vector<double> x(_GLB_N_, 1);
	vector<double> GRAD(_GLB_N_, 0.0);

	double scalar = 1.0;

	clock_t t_start_dot = clock();
	std::linalg_dot (A, B, scalar);
	double t_dot = (clock() - t_start_dot) / (double) CLOCKS_PER_SEC;

	clock_t t_start_sdot = clock();
	std::linalg_sdot(scalar, A, C);
	double t_sdot = (clock() - t_start_sdot) / (double) CLOCKS_PER_SEC;

	clock_t t_start_add = clock();
	std::linalg_add (1.0, B, 1.0, C, A);
	double t_add = (clock() - t_start_add) / (double) CLOCKS_PER_SEC;

	clock_t t_start_grad = clock();
	std::linalg_grad(_GLB_N_, _GLB_EPS_, C,  A);
	double t_grad = (clock() - t_start_grad) / (double) CLOCKS_PER_SEC;

	clock_t t_start_grad_noWriteBack = clock();
	std::linalg_grad_noWriteBack(_GLB_N_, _GLB_EPS_, C);
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
