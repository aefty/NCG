/**
 * LINEAR ALGEBRA (./cuda_sg/scrc/lib/math.cpp)
 * Math and linear alebra subrutines.
 * Refer to (./c++/lib/math.cpp) for details.
 */

namespace cpu {


   bool isNAN(double var) {
      volatile double d = var;
      return d != d;
   };

   inline void gradDiff(long int N, double EPS, vector<double>& x, vector<double>& rtrn) {
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

   inline void funGraph(long int N, double EPS, vector<double>& x, vector<long int>& C) {

      double val = 0.0;
      double EPS2 = 2.0 * EPS;
      vector<double> base(N, 0);
      cpu::gradDiff(N, EPS, x, base);

      double max_grad = *max_element(std::begin(base), std::end(base));
      double min_grad = *min_element(std::begin(base), std::end(base));
      double fuzzy = 0.5 * (max_grad + min_grad);

      for (int i = 0; i < N; i++) {
         x[i] += 1.0 ;

         for (int j = 0; j < i; j++) {
            cout << i < endl;
            val = 0.0;
            x[j] -= EPS;
            FUNCTION(N, &x[0], &val);

            val = val * -1.0;
            x[j] += EPS2;

            FUNCTION(N, &x[0], &val);

            val =  std::abs(val / EPS2  - base[j]);

            if ((val * val) > fuzzy) {
               C.push_back (j * N + i);
            }

            x[j] -= EPS2;
         }

         x[i] -= 1.0 ;
      };
   };

};

