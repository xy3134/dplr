/***********************************************************************
 *
 * NAME: lapackapi_mkl.h
 *
 * DESC: Prototypes for BLAS and LAPACK routines, definitions for
 *   real numbers, etc. for Intel's MKL.
 *
 *
 * AUTH: Yang Xinshuo
 *
 * DATE: 20140624
 *
 **********************************************************************/

#ifndef _LAPACKAPI_MKL_H_
#define _LAPACKAPI_MKL_H_
 
void dscal_(int *n, double *alpha, double *x, int *incx);
void dcopy_(int *n, double *x, int *incx, double *y, int *incy);
double dnrm2_(int *n, double *x, int *incx);
double ddot_(int *n, double *x, int *incx, double *y, int *incy);
void daxpy_(int *n, double *alpha, double *x, int *incx, double *y, int *incy);
void dgemv_(char *trans, int *m, int *n, double *alpha, double *A, int *LDA, 
			double *x, int *incx, double *beta, double *y, int *incy);
void dgemm_(char *transa, char *transb, int *m, int *n, int *k, double *alpha, 
			double *A, int *LDA, double *B, int *LDB, double *beta, double *C, int *LDC);
void dgesvd_(char *jobu, char *jobvt, int *m, int *n, double *A, int *LDA, double *S, double *U, 
			int *LDU, double *VT, int *LDVT, double *work, int *lwork, int *info);
void dger_(int *m, int *n, double *alpha, double *x, int *incx, double *y, int *incy, 
			double *A, int *LDA);
void dgetri_(int *m, double *a, int *lda, int *ipiv, double *work, int *lwork, int *info);
void dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv, int *info);


#endif
/***********************************************************************
*
* END: lapackapi_mkl.h
*
**********************************************************************/

