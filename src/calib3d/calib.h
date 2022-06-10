/* ************************************************************* calic.h *** *
 *  
 *
 * Copyright (C) 2019 Yasuyuki SUGAYA <sugaya@iim.cs.tut.ac.jp>
 *
 *                                    Time-stamp: <2019-11-12 16:12:56 sugaya>
 * ************************************************************************* */
#ifndef	__CALIB_H__
#define	__CALIB_H__

void	calc_motion (const Eigen::MatrixXd&	nV,
		     const Eigen::MatrixXd&	dV,
		     const Eigen::MatrixXd&	P,
		     Eigen::MatrixXd&		R,
		     Eigen::VectorXd&		t,
		     double			epsilon);

#endif	/* __CALIB_H__ */

/* ****************************************************** End of calib.h *** */
