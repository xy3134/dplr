/***********************************************************************
 *
 * NAME: dplr.h
 *
 * DESC: Routines for real PLR matrices
 *
 * AUTH: Yang Xinshuo
 *
 * DATE: 20140422
 *
 **********************************************************************/
#ifndef _DPLR_H_
#define _DPLR_H_

//#define USE_MKL
#define USE_NETLIB

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "dplr_internal.h"
#include "lapackapi.h"

/*
 * Initialize the conatiner for a PLR matrix. The resulting matrix is
 * "empty" in the sense that is has no size and no structure. Use
 * zplrclr() to deallocate resources used by A.
 *
 * Usage:
 *   DPlrMtx A;
 *   dplrini(&A);   // Initialize A for use
 *   ... Now you can use A ...
 *   dplrclr(&A);   // Done with A, free its resources
 */
void dplrini(DPlrMtx *A);

/*
 * Clear a PLR matrix, releasing all its resources.
 */
void dplrclr(DPlrMtx *A);

/*
 * Create a empty matrix with the same decompostion and 
 * size of the given matrix. 
 */
void dplrempty(DPlrMtx *A, DPlrMtx *B);

/*
 * Convert a dense m-by-m matrix to an m-by-m PLR matrix, B <- A, with
 * the specified number of levels and absolute accuracy eps.
 *
 * B must already be initialized (i.e. you need to call dplrini(B)).
 * It will be overwritten, and its structure changed, if necessary, to
 * match the requested nlvl.
 *
 * Both matrices are m-by-m and dense matrix A has leading dimension
 * lda >= m.
 */
void dplrd2gep(int m, double *A, int lda, DPlrMtx *B,
               double eps, int nlvl, int *info);

/*
 * Convert an m-by-m PLR matrix to a dense matrix, B <- A. ldb >= m
 * gives the leading dimension of B.
 */
void dplrgep2d(int m, DPlrMtx *A, double *B, int ldb);

/*
 * Make a copy of the m-by-m PLR matrix A, B <- A. B must already be
 * initialized.
 */
void dplrcpy(int m, DPlrMtx *A, DPlrMtx *B);


/*
 * Scale the m-by-m PLR matrix A by a scalar, i.e. 
 * A <- alpha * A.
 */
void dplrscal(int m, double *alpha, DPlrMtx *A);


/*
 * Add a scalar mutiple of a PLR matrix to another PLR matrix, i.e.
 * B <- alpha * A + B
 * where A and B are m-by-m matrices
 * alpha is a double precision scalar
 */
void dplraxpy(int m, double *alpha, DPlrMtx *A, DPlrMtx *B);

/*
 * Y <- alpha*D + Y for m-by-m PLR matrix Y, diagonal D. The diagonal
 * of D is supplied in the length-m array D.
 */
void dplradpy(int m, double *alpha, double *D,
             DPlrMtx *Y);


/*
 * Populate the array ranks with the maximum rank required at each
 * level to store the m-by-m PLR matrix A. ranks should be an array of
 * length nlvl, where nlvl is the level of PLR decomposition. On
 * entry, all elements of ranks should be initialized to -1.
 */
void dplrrnk(int m, DPlrMtx *A, int *ranks);

/*
 * Compute the total non-zero entries of a m-by-m PLR matrix
 */
long int dplrnz(int m, DPlrMtx *A);


/*
 * Reduce the ranks of the low rank blocks in the m-by-m PLR matrix A,
 * with absolute accuracy eps. 
 */
void dplrreduce(int m, DPlrMtx *A, double eps);


/*
 * Perform one of the matrix-vector operations:
 * y := alpha * A * x + beta * y or y := alpha * A' * x + beta * y
 * where x and y are vectors, alpha and beta are scalars, A is a m-by-m PLR matrix.
 */
void dplrgepmv(char trans, int m, double *alpha, DPlrMtx *A, 
	double *x, int incx, double *beta, double *y, int incy);


/*
 * C <- alpha*op(A)*op(B) + beta*C for m-by-m PLR matrix A and dense matrices B, C.
 * op(X) is X, X**T, if transX is 'N', 'T'.
 */
void dplrgepdmm(char transa, char transb, int m, int n, double *alpha, 
	DPlrMtx *A, double *B, int ldb, double *beta, double *C, int ldc);

/*
 * Performs a rank-k update to a PLR matrix A:
 * A <- A + alpha * u * s * v'
 * where  alpha  is  a scalar, u and v are m-by-k dense matrices, s is a k-by-k diagonal matrix
 * supplied as a length-k vector and A is an m-by-m PLR matrix. eps gives absolute error threshold.
 */
void dplrgeprk(int m, int k, double *alpha, double *u, int ldu, double *v, 
			int ldv, double *s, int incs, DPlrMtx *A, double eps);

/*
 * C <- alpha*op(A)*op(B) + beta*C for m-by-m PLR matrices A, B,
 * C. op(X) is X, X**T, if transX is 'N', 'T'.
 */
void dplrgeppmm(char transa, char transb, int m, double *alpha,
                DPlrMtx *A, DPlrMtx *B, double *beta,
                DPlrMtx *C, double eps);

/*
 * Invert m-by-m PLR matrix A: B <- inv(A) with absolute accuracy
 * eps. eps may be zero, but the routine will be slower because low
 * rank blocks will not be rank-reduced.
 */
void dplrgeinv(int m, DPlrMtx *A, DPlrMtx *B, double eps, int *info);


#endif

/***********************************************************************
 *
 * END: dplr.h
 *
 **********************************************************************/
