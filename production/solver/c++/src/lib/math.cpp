/**
 * LINEAR ALGEBRA (./c++/lib/math.cpp)
 * Math and linear alebra subrutines.
 */


namespace std {
  /**
   * Inner Product
   * @param A    Vector
   * @param B    Vector
   * @param rtrn Scalar
   */
  inline void linalg_dot(vector<double> const& A , vector<double> const& B, double& rtrn) {
    rtrn = 0.0;

    for (int i = 0; i < A.size(); ++i) {
      rtrn += A[i] * B[i];
    }
  };


  /**
   * Scalar Vector Product
   * @param a    Scalar
   * @param A    Vector
   * @param rtrn Scalar
   */
  inline void linalg_sdot(double const& a, vector<double> const& A ,  vector<double>&   rtrn) {
    for (int i = 0; i < A.size(); ++i) {
      rtrn[i] =  a * A[i];
    }
  };

  /**
   * Vector Vector Addition
   * @param a    Scalar
   * @param A    Vector
   * @param b    Scalar
   * @param B    Vector
   * @param rtrn Vector
   */
  inline void linalg_add(double const& a, vector<double> const& A,  double const& b, vector<double> const& B, vector<double>& rtrn) {
    for (int i = 0; i < A.size(); ++i) {
      rtrn[i] = a * A[i]  + b * B[i];
    }
  };

  /**
   * Gradient - Finite Difference
   * @param N    Probelm Size
   * @param EPS  Epsilon or dx
   * @param x    X Vector
   * @param grad Gradient Vector
   */
  inline void linalg_grad(const double N, const double EPS, vector<double> const& x, vector<double>& grad) {

    vector<double> point;
    double val = 0.0;
    double EPS2 = 2.0 * EPS;

    for (int i = 0; i < N; i++) {
      val = 0;
      point = x ;

      point[i] -= EPS;
      FUNCTION(N, &point[0], val);

      val = val * -1.0;
      point[i] += EPS2;

      FUNCTION(N, &point[0], val);
      grad[i] = (val) / (EPS2);
    }
  };


  /**
   * TESTING ONLY - Gradient function without the write back
   * @param N    Probelm Size
   * @param EPS  Epsilon or dx
   * @param x    X Vector
   */
  inline void linalg_grad_noWriteBack(const double N, const double EPS, vector<double> const& x) {

    vector<double> point;
    double val = 0.0;
    double EPS2 = 2.0 * EPS;

    for (int i = 0; i < N; i++) {
      val = 0;
      point = x ;

      point[i] -= EPS;
      FUNCTION(N, &point[0], val);

      val = val * -1.0;
      point[i] += EPS2;

      FUNCTION(N, &point[0], val);
      // grad[i] = (val) / (EPS2);
    }
  };
};

