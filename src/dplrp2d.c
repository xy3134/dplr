/***********************************************************************
 *
 * NAME: dplrp2d.c
 *
 * DESC: Convert a PLR matrix to a dense matrix.
 *
 * AUTH: Xinshuo Yang
 *
 * DATE: 20141101
 *
 ***********************************************************************/
#include "../include/dplr.h"

/*
 * Convert an m-by-m low rank matrix to a dense matrix, B <- A. ldb >= m
 * gives the leading dimension of B.
 */
static void dplrlr2d(int m, DPlrMtx *A, double *B, int ldb)
{
    if (m < 0)
    { 
        fprintf(stderr, "%s: invalid matrix size %d\n", __FUNCTION__, m);
        return;
    }
    else if (A->size != m || A->type != PLR_LRMTX)
    { 
        fprintf(stderr, "%s: invalid A type or size\n", __FUNCTION__);
        return;
    }
    else if (ldb < m)
    { 
        fprintf(stderr, "%s: invalid leading dimension\n", __FUNCTION__);
        return;
    }


    int one = 1;
    double done = 1.0, dzero = 0.0;
    char NN = 'N', TT = 'T';
    int i, j, len;

    /*
     * Quick return if possible
     */
    if (A->dlr.rank == 0)
    {
        /*
         * Zero out B
         */
        for (i = 0; i < m; ++i)
            for (j = 0; j < m; ++j)
                B[i*ldb+j] = 0.0;
        return;
    }


    /*
     * Compute B <- u * s * v'
     */

    len = m*A->dlr.rank;
    double *temp;
    temp = malloc(m*A->dlr.rank*sizeof(double));

    /*
     * temp <- u
     */
    dcopy_(&len, A->dlr.u, &one, temp, &one);

    /*
     * temp <- temp * s
     */
    for (i = 0; i < A->dlr.rank; ++i)
    {
        dscal_(&m, A->dlr.sig+i, temp+i*m, &one);
    }

    /*
     * B <- temp * v'
     */
    dgemm_(&NN, &TT, &m, &m, &A->dlr.rank, &done, temp, &m, A->dlr.v, &m, &dzero, B, &ldb);

    /*
     * Free memory
     */
    free(temp);

    /*
     * Success
     */
    return;
}

/*
 * Convert an m-by-m PLR matrix to a dense matrix, B <- A. ldb >= m
 * gives the leading dimension of B.
 */
void dplrgep2d(int m, DPlrMtx *A, double *B, int ldb)
{
    if (m < 0)
    { 
        fprintf(stderr, "%s: invalid matrix size %d\n", __FUNCTION__, m);
        return;
    }
    else if (A->size != m)
    { 
        fprintf(stderr, "%s: invalid A size %d\n", __FUNCTION__, m);
        return;
    }
    else if (A->type != PLR_DMTX && A->type != PLR_HMTX)
    { 
        fprintf(stderr, "%s: invalid A type\n", __FUNCTION__);
        return;
    }
    else if (ldb < m)
    { 
        fprintf(stderr, "%s: invalid leading dimension\n", __FUNCTION__);
        return;
    }

    int one = 1;
    int i;

    if (A->type == PLR_DMTX)
    {
        for (i = 0; i < m; ++i)
            dcopy_(&m, A->dd.d+i*m, &one, B+i*ldb, &one);
    }
    else
    {   
        /*
         * Recover low rank matrices
         */
        dplrlr2d(m/2, A->dh.ne, B+ldb*m/2, ldb);
        dplrlr2d(m/2, A->dh.sw, B+m/2, ldb);

        /*
         * Recover diagonal blocks recursively
         */
        dplrgep2d(m/2, A->dh.nw, B, ldb);
        dplrgep2d(m/2, A->dh.se, B+ldb*m/2+m/2, ldb);
    }
}

/***********************************************************************
 *
 * END: dplrp2d.c
 *
 **********************************************************************/


