/* **************************************************************** ls.c *** *
 *  
 *
 * Copyright (C) 2019 Yasuyuki SUGAYA <sugaya@iim.cs.tut.ac.jp>
 *
 *                                    Time-stamp: <2019-11-21 16:56:28 sugaya>
 * ************************************************************************* */
#include <stdio.h>
#include <stdlib.h>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <exception>

using namespace std;
double calc_motion(const Eigen::MatrixXd &nV,
                   const Eigen::MatrixXd &dV,
                   const Eigen::MatrixXd &P,
                   Eigen::MatrixXd &R,
                   Eigen::VectorXd &t,
                   double epsilon)
{
  int nlines = nV.cols();
  double err = numeric_limits<double>::infinity();

  std::vector<Eigen::MatrixXd> K;
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(3, 3);
  for (int n = 0; n < nlines; n++)
  {
    Eigen::MatrixXd K_ = I - (nV.col(n) * nV.col(n).transpose());
    K.push_back(K_);
  }

  Eigen::MatrixXd A(3, nlines);
  int count = 1;

  for (;;)
  {

    for (int n = 0; n < nlines; n++)
      A.col(n) = K[n] * R * dV.col(n);

    Eigen::MatrixXd M = A * dV.transpose();
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M,
                                          Eigen::ComputeThinU |
                                              Eigen::ComputeThinV);
    Eigen::MatrixXd E = svd.singularValues().asDiagonal();

    int rank = 0;
    for (int n = 0; n < 3; n++)
    {
      if (fabs(E(n, n)) > epsilon)
        rank++;
    }
    Eigen::MatrixXd S = Eigen::MatrixXd::Identity(3, 3);

    if (rank == 3)
    {
      if (M.determinant() < 0.0)
      {
        S(2, 2) = -1;
      }
    }
    else if (rank == 2)
    {
      if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0)
      {
        S(2, 2) = -1;
      }
    }
    else
    {
      // fprintf (stderr, "The rank of the matrix is less then 2.\n");
      throw std::runtime_error("The rank of the matrix is less then 2.\n");
      return;
    }
    Eigen::MatrixXd R_ = svd.matrixU() * S * svd.matrixV().transpose();

    err = fabs((R - R_).norm());
    // fprintf(stderr, "%d %.16e\n", count++, err);

    if (err < epsilon)
      break;

    R = R_;
  }
  Eigen::MatrixXd IK = Eigen::MatrixXd::Zero(3, 3);
  for (int n = 0; n < nlines; n++)
  {
    IK += (nV.col(n) * nV.col(n).transpose());
  }
  Eigen::MatrixXd IKinv = IK.inverse();
  Eigen::VectorXd K_IRP = Eigen::VectorXd::Zero(3);
  for (int n = 0; n < nlines; n++)
  {
    K_IRP += ((K[n] - I) * R * P.col(n));
  }
  t = IKinv * K_IRP;
  return err;
}

/* ********************************************************* End of ls.c *** */
