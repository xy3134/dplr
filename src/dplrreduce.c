/***********************************************************************
 *
 * NAME: dplrreduce.c
 *
 * DESC: Reduce the rank of a PLR matrix
 *
 * AUTH: Xinshuo Yang
 *
 * DATE: 20141224
 *
 ***********************************************************************/
#include "../include/dplr.h"
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

/*
 * Fortran90 subroutines that perform Gram-Schmidt orthognalization.
 */
extern void dortlr_(int *, int *, double *, double *, int *, double *, int *, double *, int *);
extern void ortlrini_(int *, int *);

/*
 * Reduce the ranks of the low rank blocks in the m-by-m PLR matrix A,
 * with absolute accuracy eps. 
 */
void dplrreduce(int m, DPlrMtx *A, double eps)
{
	if (m < 0)
	{
		fprintf(stderr, "%s: invalid matrix size %d\n", __FUNCTION__, m);
		return;	
	}
	if (A->type != PLR_DMTX && A->type != PLR_HMTX)
	{
		fprintf(stderr, "%s: invalid A type\n", __FUNCTION__);
		return;	
	}
	else if (A->size != m)
	{
		fprintf(stderr, "%s: invalid A size\n", __FUNCTION__);
		return;
	}

	/*
	 * Number of iteration for Gram-Schmit process.
	 */
	int niter = 1;

	if (A->type == PLR_DMTX)
	{
		/*
		 * Dense matrix, no reduction needed
		 */
		;
	}
	else
	{
		double eps2 = eps/4.0;

		int len = m/2, *ipivot;
		ipivot = malloc(MAX(A->dh.ne->dlr.rank, A->dh.sw->dlr.rank)*sizeof(int));

		ortlrini_(&A->dh.ne->dlr.rank, ipivot);
		dortlr_(&len, &len, A->dh.ne->dlr.u, A->dh.ne->dlr.v, &A->dh.ne->dlr.rank,
			A->dh.ne->dlr.sig, ipivot, &eps2, &niter);

		ortlrini_(&A->dh.sw->dlr.rank, ipivot);
		dortlr_(&len, &len, A->dh.sw->dlr.u, A->dh.sw->dlr.v, &A->dh.sw->dlr.rank, 
			A->dh.sw->dlr.sig, ipivot, &eps2, &niter);

		/*
		 * Free memory
		 */
		free(ipivot);

		/*
		 * Reduce the rank recursively
		 */
		dplrreduce(len, A->dh.nw, eps2);
		dplrreduce(len, A->dh.se, eps2);
	}
}

/***********************************************************************
 *
 * END: dplrreduce.c
 *
 **********************************************************************/
