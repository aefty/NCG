namespace std {

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

   /*
      inline void linalg_grad(long int N,  double EPS, vector<double> & x, vector<double>& grad) {

         vector<double> point;
         double* val;
         double EPS2 = 2.0 * EPS;

         for (int i = 0; i < N; i++) {
            val[0] = 0;
            point = x ;

            point[i] -= EPS;
            FUNCTION(N, &point[0], val);

            val[0] = val[0] * -1.0;
            point[i] += EPS2;

            FUNCTION(N, &point[0], val);

            grad[i] = (val[0]) / (EPS2);
         }
      };

      inline void test_NfunCall( long int N, vector<double> & x) {

         vector<double> point;
         double* val;

         for (int i = 0; i < N; i++) {
            val[0] = 0;
            point = x ;
            FUNCTION(N, &point[0], val);
         }
      };
   */
};

