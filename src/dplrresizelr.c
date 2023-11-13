/***********************************************************************
 *
 * NAME: dplrresizelr.c
 *
 * DESC: Increase the capacity of a low rank matrix
 *
 * AUTH: Yang Xinshuo
 *
 * DATE: 20140425
 *
 **********************************************************************/
#include "../include/dplr.h"


void dplrresizelr(DPlrMtx *A, int capacity)
{
    int one=1;

    int j;
    double nanv;
    
    int len;
    double *u, *v;
    double *sig;

    if (A->type != PLR_LRMTX)
    {
        fprintf(stderr, "%s: invalid matrix type\n", __FUNCTION__);
        return;
    }
    else if (A->size < 0)
    {
        fprintf(stderr, "%s: invalid matrix size\n", __FUNCTION__);
        return;
    }
    else if (capacity < 0)
    {
        fprintf(stderr, "%s: invalid capacity\n", __FUNCTION__);
        return;
    }

    if (capacity < A->dlr.capacity)
    {
        /*
         * Do nothing if matrix already has sufficient capacity
         */
        return;
    }

    /*
     * We always at least double the capacity, to minimize the number
     * of times we have to resize
     */
    capacity = capacity > 2*A->dlr.capacity ? capacity : 2*A->dlr.capacity;

    /*
     * Allocate new memory
     */
    u   = malloc(capacity*A->size*sizeof(double));
    v   = malloc(capacity*A->size*sizeof(double));
    sig = malloc(capacity*sizeof(double));

    if (!u || !v || !sig)
    {
        fprintf(stderr, "%s: memory allocation failed\n", __FUNCTION__);
        return;
    }

    nanv = nan("");
    for (j = 0; j < capacity*A->size; ++j) 
    {
        u[j] = nanv;
    }
    for (j = 0; j < capacity*A->size; ++j) 
    {
        v[j] = nanv;
    }
    for (j = 0; j < capacity; ++j) 
    {
        sig[j] = nanv;
    }

    /*
     * Copy old data, if needed
     *
     * NOTE: We don't use realloc() above because in many cases, only
     *   a faction of the total space is currently being used (rank <<
     *   capacity), so it is wasteful to copy the unusued
     *   data. (rdl20140430)
     */
    if (A->dlr.rank > 0)
    {
        dcopy_(&A->dlr.rank, A->dlr.sig, &one, sig, &one);
        len = A->dlr.rank*A->size;
        dcopy_(&len, A->dlr.u, &one, u, &one);
        dcopy_(&len, A->dlr.v, &one, v, &one);
    }

    /*
     * Release old buffers and update with new ones
     */
    if (A->dlr.capacity > 0)
    {
        free(A->dlr.u);
        free(A->dlr.v);
        free(A->dlr.sig);
    }

    A->dlr.capacity = capacity;
    A->dlr.u   = u;
    A->dlr.v   = v;
    A->dlr.sig = sig;

    /*
     * Success
     */
    return;
}

/***********************************************************************
 *
 * END: dplrresizelr.c
 *
 **********************************************************************/
