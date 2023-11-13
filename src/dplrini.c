/***********************************************************************
 *
 * NAME: dplrini.c
 *
 * DESC: Initialize a PLR matrix.
 *
 * AUTH: Yang Xinshuo
 *
 * DATE: 20140424
 *
 **********************************************************************/
#include "../include/dplr.h"


/*
 * Initialize matrix with unspecified type and size
 */
void dplrini(DPlrMtx *A)
{
    /*
     * Zero all fields, set the type as uninitialized
     */
    memset(A, 0, sizeof(DPlrMtx));
    A->type = PLR_UMTX;

    /*
     * Success
     */
    return;
}

/*
 * Initialize dense matrix
 */
void dplrinid(int m, DPlrMtx *A)
{
    int j;
    double nanv;

    if (m < 0)
    {
        fprintf(stderr, "%s: invalid matrix size\n", __FUNCTION__);
        return;
    }

    A->type = PLR_DMTX;
    A->size = m;
    if (m > 0)
    {
        A->dd.d = malloc(m*m*sizeof(double));

        if (!A->dd.d)
        {
            fprintf(stderr, "%s: memory allocation failed\n", __FUNCTION__);
            return;
        }

        nanv = nan("");
        for (j = 0; j < m*m; ++j) 
        {
            A->dd.d[j] = nanv;
        }

    }
    else
    {
        A->dd.d = 0;
    }

    /*
     * Success
     */
    return;
}


/*
 * Initialize low rank matrix with specified capacity
 */
void dplrinilr(int m, DPlrMtx *A, int capacity)
{
    int j;
    double nanv;

    if (m < 0 || capacity < 0)
    {
        fprintf(stderr, "%s: invalid size or capacity\n", __FUNCTION__);
        return;
    }

    /*
     * Initialize fields and allocate storage if requested
     */
    A->type         = PLR_LRMTX;
    A->size         = m;
    A->dlr.rank     = 0;
    A->dlr.capacity = capacity;
    if (capacity > 0)
    {
        A->dlr.u   = malloc(m*capacity*sizeof(double));
        A->dlr.sig = malloc(capacity*sizeof(double));
        A->dlr.v   = malloc(m*capacity*sizeof(double));
        if (!A->dlr.u || !A->dlr.sig || !A->dlr.v)
        {
            fprintf(stderr, "%s: memory allocation failed\n", __FUNCTION__);
            return;
        }

        nanv = nan("");
        for (j = 0; j < m*capacity; ++j) 
        {
            A->dlr.u[j] = nanv;
        }
        for (j = 0; j < m*capacity; ++j) 
        {
            A->dlr.v[j] = nanv;
        }
        for (j = 0; j < capacity; ++j) 
        {
            A->dlr.sig[j] = nanv;
        }

    }
    else
    {
        A->dlr.u = A->dlr.v = 0;
        A->dlr.sig = 0;
    }

    /*
     * Success
     */
    return;
}


/*
 * Initialize hierarchical matrix. Submatrices are initialized with
 * zhini(), so they do not have valid sizes or types. You have to do
 * that yourself.
 */
void dplrinih(int m, DPlrMtx *A)
{
    if (m < 0)
    {
        fprintf(stderr, "%s: invalid size\n", __FUNCTION__);
        return;
    }
    else if (m % 2 != 0)
    {
        fprintf(stderr, "%s: m must be even\n", __FUNCTION__);
        return;
    }

    A->type  = PLR_HMTX;
    A->size  = m;
    A->dh.nw = malloc(sizeof(DPlrMtx));
    A->dh.ne = malloc(sizeof(DPlrMtx));
    A->dh.se = malloc(sizeof(DPlrMtx));
    A->dh.sw = malloc(sizeof(DPlrMtx));
    if (!A->dh.nw || !A->dh.ne || !A->dh.se || !A->dh.sw)
    {
        fprintf(stderr, "%s: memory allocation failed\n", __FUNCTION__);
        return;
    }
    dplrini(A->dh.nw);
    dplrini(A->dh.ne);
    dplrini(A->dh.se);
    dplrini(A->dh.sw);

    /*
     * Success
     */
    return;
}




/*
 * Deallocate memories for PLR matrix A
 */
void dplrclr(DPlrMtx *A)
{
    if (A->type == PLR_UMTX)
    {
        ;
    }
    else if (A->type == PLR_DMTX)
    {
        free(A->dd.d);
    }
    else if (A->type == PLR_LRMTX)
    {
        free(A->dlr.u);
        free(A->dlr.v);
        free(A->dlr.sig);
    }
    else
    {
        dplrclr(A->dh.nw);
        dplrclr(A->dh.se);
        dplrclr(A->dh.ne);
        dplrclr(A->dh.sw);
    }
}



/***********************************************************************
 *
 * END: dplrini.c
 *
 **********************************************************************/
