/***********************************************************************
 *
 * NAME: dplraxpy.c
 *
 * DESC: Adds a scalar multiple of a PLR matrix to another PLR matrix
 *
 * AUTH: Yang Xinshuo
 *
 * DATE: 20141103
 *
 ***********************************************************************/
#include "../include/dplr.h"

/*
 * Compute B <- alpha * A + B, A and B are PLR low rank matices.
 */
static void dplrlraxpy(int m, double *alpha, DPlrMtx *A, DPlrMtx *B)
{
    if (m < 0)
    {
        fprintf(stderr, "%s: invalid matrix size %d\n", __FUNCTION__, m);
        return;
    }   
    if (A->type != PLR_LRMTX || B->type != PLR_LRMTX)
    {
        fprintf(stderr, "%s: invalid matrix type\n", __FUNCTION__);
        return;
    }
    else if (A->size != m || B->size != m)
    {
        fprintf(stderr, "%s: invalid A or B size\n", __FUNCTION__);
        return;
    }

    int i, len, one = 1;
    double dnone = -1.0, fff = fabs(*alpha);

    /*
     * Quick return if possible
     */
    if (A->dlr.rank == 0)
        return;

    /*
     * Increase the capacity of B if necessary
     */
    if (B->dlr.capacity < A->dlr.rank+B->dlr.rank)  
        dplrresizelr(B, A->dlr.rank+B->dlr.rank);

    /*
     * Copy alpha*A to B
     */
    len = m*A->dlr.rank;
    dcopy_(&len, A->dlr.u, &one, B->dlr.u+B->dlr.rank*m, &one);
    dcopy_(&len, A->dlr.v, &one, B->dlr.v+B->dlr.rank*m, &one);

    for (i = 0; i < A->dlr.rank; ++i)
        B->dlr.sig[B->dlr.rank+i] = fff*A->dlr.sig[i];

    /*
     * if alpha < 0.0, negative u-vectors copied from A
     */
    if (*alpha < 0.0)
    {
        dscal_(&len, &dnone, B->dlr.u+B->dlr.rank*m, &one);
    }

    /*
     * Adjust rank of Y
     */
    B->dlr.rank += A->dlr.rank;
}


/*
 * Perform the following operation:
 * B <- alpha * A + B
 * where A and B are m-by-m PLR matrices
 * alpha is a double precision scalar
 */
void dplraxpy(int m, double *alpha, DPlrMtx *A, DPlrMtx *B)
{
    if (m < 0)
    {
        fprintf(stderr, "%s: invalid matrix size %d\n", __FUNCTION__, m);
        return;   
    }
    else if (A->size != m || B->size != m)
    {
        fprintf(stderr, "%s: invalid A or B size\n", __FUNCTION__);
        return;
    }
    else if (A->type != B->type)
    {
        fprintf(stderr, "%s: A and B types do not match\n", __FUNCTION__);
        return;
    }
    else if (A->type != PLR_DMTX && A->type != PLR_HMTX)
    {
        fprintf(stderr, "%s: invalid A or B type\n", __FUNCTION__);
        return;   
    }

    /*
     * if alpha = 0
     */
    if (*alpha == 0)
        return;


    int one = 1;
    int len;
    

    if (A->type == PLR_DMTX)
    {
        len = m*m;
        daxpy_(&len, alpha, A->dd.d, &one, B->dd.d, &one);
    }
    else
    {
        /*
         * Adds off-diagonal blocks
         */
        dplrlraxpy(m/2, alpha, A->dh.ne, B->dh.ne);
        dplrlraxpy(m/2, alpha, A->dh.sw, B->dh.sw);

        /*
         * Adds alpha * A recursively
         */
        dplraxpy(m/2, alpha, A->dh.nw, B->dh.nw);
        dplraxpy(m/2, alpha, A->dh.se, B->dh.se);
    }
}


/*
 * B <- alpha * D + B for m-by-m PLR matrix B, diagonal D. The diagonal
 * of D is supplied in the length-m array D.
 */
void dplradpy(int m, double *alpha, double *D, DPlrMtx *B)
{
    if (m < 0)
    {
        fprintf(stderr, "%s: invalid matrix size %d\n", __FUNCTION__, m);
        return;   
    }
    else if (B->type != PLR_DMTX && B->type != PLR_HMTX)
    {
        fprintf(stderr, "%s: invalid B type\n", __FUNCTION__);
        return;
    }
    else if (B->size != m)
    {
        fprintf(stderr, "%s: invalid B size\n", __FUNCTION__);
        return; 
    }

    int i;

    if (B->type == PLR_DMTX)
    {
        /*
         * Add diagonal matrix D to dense matrix
         */
        for (i = 0; i < m; ++i)
        {
            B->dd.d[i*m+i] += (*alpha)*D[i];
        }
    }
    else
    {
        /*
         * Add diagonal matrix recursiely.
         */
        dplradpy(m/2, alpha, D, B->dh.nw);
        dplradpy(m/2, alpha, D+m/2, B->dh.se);
    }

}

/***********************************************************************
 *
 * END: dplraxpy.c
 *
 ***********************************************************************/

