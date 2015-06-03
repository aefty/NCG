namespace std {
   inline void linalg_dot(vector<double> const& A , vector<double> const& B, double& rtrn) {
      rtrn = 0.0;

      for (int i = 0; i < A.size(); ++i) {
         rtrn += A[i] * B[i];
      }
   };

   inline void linalg_sdot(double const& a, vector<double> const& A ,  vector<double>&   rtrn) {
      for (int i = 0; i < A.size(); ++i) {
         rtrn[i] =  a * A[i];
      }
   };

   inline void linalg_add(double const& a, vector<double> const& A,  double const& b, vector<double> const& B, vector<double>& rtrn) {
      for (int i = 0; i < A.size(); ++i) {
         rtrn[i] = a * A[i]  + b * B[i];
      }
   };

   inline void linalg_grad(const double N, const double EPS, vector<double> const& x, vector<double>& grad) {

      vector<double> point;
      double val = 0.0;
      double EPS2 = 2.0 * EPS;

      for (int i = 0; i < N; i++) {
         val = 0;
         point = x ;

         point[i] -= EPS;
         FUNCTION(&point[0], val);

         val = val * -1.0;
         point[i] += EPS2;

         FUNCTION(&point[0], val);
         grad[i] = (val) / (EPS2);
      }
   };

   inline void test_NfunCall(const double N, vector<double> const& x) {

      vector<double> point;
      double val = 0.0;

      for (int i = 0; i < N; i++) {
         val = 0;
         point = x ;
         FUNCTION(&point[0], val);
      }
   };
};

