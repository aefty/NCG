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
};

