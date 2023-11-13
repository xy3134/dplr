/***********************************************************************
 *
 * NAME: dplrd2p.c
 *
 * DESC: Compress a dense matrix to PLR
 *
 * AUTH: Yang Xinshuo
 *
 * DATE: 20140422
 *
 **********************************************************************/
#include "../include/dplr.h"

#define MAX(x,y) ((x)>(y) ? (x) : (y))


/*
 * Use the SVD to get a low rank approximation of A with absolute
 * accuracy eps, and store the result in B. Vectors will be
 * normalized.
 */
static void dplrlrappx(int m, double *A, int lda, DPlrMtx *B,
                      double eps, int *info)
{
    static int one=1, mone=-1;
    static char OO='O', AA='A';

    double *U, *Vh, workq, *work;
    double *sig;
    int j, rank, lwork, len;

    *info = -1;
    if (m < 0)
    { 
        fprintf(stderr, "%s: invalid matrix size %d\n", __FUNCTION__, m);
        return;
    }
    else if (lda < m)
    {
        fprintf(stderr, "%s: invalid leading dimension\n", __FUNCTION__);
        return;
    }
    else if (eps < 0.0)
    {
        fprintf(stderr, "%s: eps must be nonnegative\n", __FUNCTION__);
        return;
    }
    else if (B->type != PLR_LRMTX || B->size != m)
    {
        fprintf(stderr, "%s: invalid B type or size\n", __FUNCTION__);
        return;
    }

    /*
     * Compute the SVD of the input matrix
     */
    U   = malloc(m*m*sizeof(double));
    Vh  = malloc(m*m*sizeof(double));
    sig = malloc(  m*sizeof(double));
    if (!U || !sig || !Vh)
    {
        fprintf(stderr, "%s: memory allocation failed\n", __FUNCTION__);
        if (U)   free(U);
        if (sig) free(sig);
        if (Vh)  free(Vh);
        *info = -1;
        return;
    }
    
    
    for (j = 0; j < m; ++j) 
        dcopy_(&m, (double *)A + j*lda, &one, U + j*m, &one);
/*
 * FIXME: zgesdd() fails on some older version of MKL. For now, we use
 *   the slower zgesvd() until it is possible to upgrade.
 *   (rdl20140702)
 *
 *   zgesdd_(&OO, &m, &m, U, &m, sig, 0, &one, Vh, &m, 
 *           &workq, &mone, rwork, iwork, info);
 */
    dgesvd_(&OO, &AA, &m, &m, U, &m, sig, 0, &one, Vh, &m,
            &workq, &mone, info);
    if (*info != 0)
    {
        fprintf(stderr, "%s: dgesvd() workspace query failed: %d\n",
                __FUNCTION__, *info);
        free(U);
        free(sig);
        free(Vh);
        return;
    }

    lwork = (int)(workq);
    work  = malloc(lwork*sizeof(double));
    if (!work)
    {
        fprintf(stderr, "%s: memory allocation failed\n", __FUNCTION__);
        free(U);
        free(sig);
        free(Vh);
        *info = -1;
        return;
    }
/*
 * FIXME: See note above about zgesdd() vs. zgesvd(). (rdl20140702)
 *
 *  zgesdd_(&OO, &m, &m, U, &m, sig, 0, &one, Vh, &m, 
 *          work, &lwork, rwork, iwork, info);
 */
    dgesvd_(&OO, &AA, &m, &m, U, &m, sig, 0, &one, Vh, &m,
            work, &lwork, info);
    if (*info != 0)
    {
        fprintf(stderr, "%s: dgesvd() failed: %d\n", __FUNCTION__, *info);
        free(U);
        free(sig);
        free(Vh);
        free(work);
        return;
    }
    

    /*
     * Determine the rank of the approximation
     */
    for (rank = 0; rank < m; ++rank) 
        if (sig[rank] < eps) break;

    /*
     * Increase the capacity of B, if necessary
     */
    if (B->dlr.capacity < rank) dplrresizelr(B, rank);

    /*
     * Store the approximation
     */
    B->dlr.rank = rank;

    /*
     * If the rank was 0, we're done
     */
    if (rank == 0)
    {
        free(U);
        free(sig);
        free(Vh);
        free(work);
        *info = 0;
        return;
    }

    /*
     * Save the singular values and vectors
     */
    dcopy_(&rank, sig, &one, B->dlr.sig, &one);

    /* Copy U as is */
    len = rank*m;
    dcopy_(&len, U, &one, B->dlr.u, &one);

    /* Need conjugate transpose of V */
    for (j = 0; j < rank; ++j)
    {
        dcopy_(&m, Vh + j, &m, B->dlr.v + j*m, &one);
    }


    /*
     * Success
     */
    free(U);
    free(sig);
    free(Vh);
    free(work);

    *info = 0;
    return;
}


/*
 * Compress dense matrix to a PLR matrix with the specificed
 * number of levels (nlvl == 0 means dense matrix, 1 means 2-by-2,
 * etc.).  eps gives absolute error threshold.
 */
void dplrd2gep(int m, double *A, int lda, DPlrMtx *B,
              double eps, int nlvl, int *info)
{
    int one=1;

    double eps2;
    int j;

    *info = -1;
    if (m < 0)
    { 
        fprintf(stderr, "%s: invalid matrix size %d\n", __FUNCTION__, m);
        return;
    }
    else if (lda < m)
    {
        fprintf(stderr, "%s: invalid leading dimension\n", __FUNCTION__);
        return;
    }
    else if (eps < 0.0)
    {
        fprintf(stderr, "%s: eps must be nonnegative\n", __FUNCTION__);
        return;
    }

    /*
     * Does the requested decomposition level make sense?
     */
    if (m % (1<<nlvl) != 0)
    {
        fprintf(stderr, "%s: m not divisible by 2**nlvl\n", __FUNCTION__);
        return;
    }

    /*
     * Are we making a dense matrix?
     */
    if (nlvl == 0)
    {
        /*
         * Is B already a dense matrix of the correct type and size?
         */
        if (B->type != PLR_DMTX || B->size != m)
        {
            dplrclr(B);
            dplrinid(m, B);
        }

        /*
         * Do the copy
         */
        for (j = 0; j < m; ++j) 
            dcopy_(&m, (double *)A + j*lda, &one, B->dd.d + j*m, &one);
    }
    else
    {
        /*
         * Form a 2-by-2 block matrix at this level. There are 2 cases:
         *
         *   (1) This is a level-1 decomposition, so the off-diag blocks
         *       are low rank approximations and the diag blocks are dense.
         *
         *   (2) This is a level-2-or-greater decomposition to the off-diag
         *       blocks are low rank approximations and the diag blocks are
         *       PLR approximations.
         *
         * This matters because it determines how we allocate the error
         * to each block of the 2-by-2 block matrix we construct.
         */
        if (nlvl == 1)
        {
            /*
             * This is the last level of decomposition, we we allocate
             * the error to each of the 2 low rank off-diag blocks. The
             * diag blocks are dense, and therefore do not have any 
             * approximation error.
             */
            eps2 = eps/2.0;
        }
        else
        {
            /*
             * There will be more levels of decomposition, so allocate
             * the error equally to all 4 blocks.
             */
            eps2 = eps/4.0;
        }

        /*
         * Make sure this matrix is of the proper structure
         */
        if (B->type != PLR_HMTX || B->size != m)
        {
            dplrclr(B);
            dplrinih(m, B);
            dplrinilr(m/2, B->dh.ne, 0);
            dplrinilr(m/2, B->dh.sw, 0);
            /* 
             * Leave diag blocks uninitialized -- recursive call will
             * initialize them
             */
        }

        /*
         * Construct off-diagonal low-rank approximations
         */
        dplrlrappx(m/2, A + m/2*lda, lda, B->dh.ne, eps2, info);
        if (*info != 0)
        {
            fprintf(stderr, "%s: failed to approximate NE block\n", 
                    __FUNCTION__);
            return;
        }
        dplrlrappx(m/2, A + m/2, lda, B->dh.sw, eps2, info);
        if (*info != 0)
        {
            fprintf(stderr, "%s: failed to approximate SW block\n", 
                    __FUNCTION__);
            return;
        }
        
        /*
         * Construct diagonal blocks recursively
         */
        dplrd2gep(m/2, A, lda, B->dh.nw, eps2, nlvl-1, info);
        if (*info != 0)
        {
            fprintf(stderr, "%s: failed to approximate NW block\n", 
                    __FUNCTION__);
            return;
        }
        dplrd2gep(m/2, A + m/2 + m/2*lda, lda, B->dh.se, eps2, nlvl-1, info);
        if (*info != 0)
        {
            fprintf(stderr, "%s: failed to approximate SE block\n", 
                    __FUNCTION__);
            return;
        }
    }

    /*
     * Success
     */
    *info = 0;
    return;
}

/***********************************************************************
 *
 * END: dplrd2p.c
 *
 **********************************************************************/
