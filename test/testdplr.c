#define USE_NETLIB
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "../include/lapackapi.h"
#include "../include/dplr.h"

int main()
{
	char NN = 'N';
	int one = 1, mone = -1;
	double dzero = 0.0, done = 1.0;
	double eps = pow(10.0, -6.0);
	clock_t t0, t1;

	int m, nlvl, info, i, j, len, lwork;
	double err;

	int *ipiv;
	double *work, *A, *B, *C, tmp;
	double *A_approx, *B_approx, *C_approx;

	DPlrMtx *PA, *PB, *PC;

	m = 2048;
	len = m * m;
	ipiv = malloc(len * sizeof(int));
	A = malloc(len * sizeof(double));
	B = malloc(len * sizeof(double));
	C = malloc(len * sizeof(double));
	A_approx = malloc(len * sizeof(double));
	B_approx = malloc(len * sizeof(double));
	C_approx = malloc(len * sizeof(double));

	PA = malloc(sizeof(DPlrMtx));
	PB = malloc(sizeof(DPlrMtx));
	PC = malloc(sizeof(DPlrMtx));

	/*
	 * Create a dense matrix A 
	 */
	for (i = 0; i < m; ++ i){
		for (j = 0; j < m; ++j){
			if (i == j){
				A[i * m + j] = 0.0;
			}
			else {
				A[i * m + j] = 1. / abs(i - j);
			}
		}
	}

	/*
	 * Compute B <- A * A and C <- A^-1
	 */
	t0 = clock(); 
	dgemm_(&NN, &NN, &m, &m, &m, &done, A, &m, A, &m, &dzero, B, &m);
	t1 = clock();
	printf("dense matrix multiplication took %f seconds\n", 
		((double)(t1 - t0))/CLOCKS_PER_SEC);

	dcopy_(&len, A, &one, C, &one);
	t0 = clock(); 
	dgetrf_(&m, &m, C, &m, ipiv, &info);
	if (info != 0){
		printf("LU factorization failed in dgetrf!\n");
		return -1;
	}
	dgetri_(&m, C, &m, ipiv, &tmp, &mone, &info);
	if (info != 0){
		printf("Matrix inverse optimal LWORK query failed in dgetri!\n");
		return -1;
	}
	lwork = (int)tmp;
	work = malloc(lwork * sizeof(double));
	dgetri_(&m, C, &m, ipiv, work, &lwork, &info);
	if (info != 0){
		printf("Matrix ivnerse failed in dgetri!\n");
		return -1;
	}
	t1 = clock();
	printf("dense matrix invesion took %f seconds\n", 
		((double)(t1 - t0))/CLOCKS_PER_SEC);

	/*
	 * Convert A to PLR
	 */
	dplrini(PA);
	dplrini(PB);
	dplrini(PC);

	nlvl = 6;
	dplrd2gep(m, A, m, PA, eps, nlvl, &info);
	if (info != 0){
		printf("Compression of A into PLR form failed!\n");
		return -1;
	}
	dplrgep2d(m, PA, A_approx, m);

	/*
	 * Compute PB <- PA * PA
	 */
	t0 = clock(); 
	dplrempty(PA, PB);
	dplrgeppmm(NN, NN, m, &done, PA, PA, &dzero, PB, eps);
	dplrgep2d(m, PB, B_approx, m);
	t1 = clock();
	printf("PLR matrix multiplication took %f seconds\n", 
		((double)(t1 - t0))/CLOCKS_PER_SEC);

	/*
	 * Compute inverse of PA
	 */
	t0 = clock(); 
	dplrgeinv(m, PA, PC, eps, &info);
	if (info != 0){
		printf("PLR inversion failed!!!\n");
		return -1;
	}
	dplrgep2d(m, PC, C_approx, m);
	t1 = clock();
	printf("PLR matrix invesion took %f seconds\n", 
		((double)(t1 - t0))/CLOCKS_PER_SEC);

	/*
	 * Error estimation
	 */
	err = 0.;
	for (i = 0; i < len; ++i){
		err += (A[i] - A_approx[i]) * (A[i] - A_approx[i]);
	}
	err = sqrt(err);
	printf("Matrix compression error is %e\n", err);

	err = 0.;
	for (i = 0; i < len; ++i){
		err += (B[i] - B_approx[i]) * (B[i] - B_approx[i]);
	}
	err = sqrt(err);
	printf("Matrix multiplication error is %e\n", err);

	err = 0.;
	for (i = 0; i < len; ++i){
		err += (C[i] - C_approx[i]) * (C[i] - C_approx[i]);
	}
	err = sqrt(err);

	printf("Matrix inversion error is %e\n", err);
	
	free(work);
	free(ipiv);
	free(A);
	free(B);
	free(C);
	free(A_approx);
	free(B_approx);
	free(C_approx);
	dplrclr(PA);
	dplrclr(PB);
	dplrclr(PC);

	return 0;
}
