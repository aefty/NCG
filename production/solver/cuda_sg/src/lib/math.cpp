/**
 * LINEAR ALGEBRA (./cuda_sg/scrc/lib/math.cpp)
 * Math and linear alebra subrutines.
 * Refer to (./c++/lib/math.cpp) for details.
 */

namespace cpu {

   inline void linalg_dot(vector<double>& A , vector<double>& B, double& rtrn) {
      rtrn = 0.0;

      for (int i = 0; i < A.size(); ++i) {
         rtrn += A[i] * B[i];
      }
   };

   inline void linalg_sdot(const double& a, vector<double>& A ,  vector<double>&   rtrn) {
      for (int i = 0; i < A.size(); ++i) {
         rtrn[i] =  a * A[i];
      }
   };

   inline void linalg_add(const double& a, vector<double>& A,  const double& b, vector<double>& B, vector<double>& rtrn) {
      for (int i = 0; i < A.size(); ++i) {
         rtrn[i] = a * A[i]  + b * B[i];
      }
   };

   inline void linalg_grad(long int N, double EPS, vector<double> const& x, vector<double>& grad) {

      double val = 0.0;
      double EPS2 = 2.0 * EPS;

      for (int i = 0; i < N; i++) {
         val = 0;

         x[i] -= EPS;
         FUNCTION(N, &x[0], &val);

         val = val * -1.0;
         x[i] += EPS2;

         FUNCTION(N, &x[0], &val);
         rtrn[i] = (val) / (EPS2);
         x[i] -= EPS2;
      }
   };
};

