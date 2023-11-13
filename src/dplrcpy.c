/***********************************************************************
 *
 * NAME: dplrcpy.c
 *
 * DESC: Make a copy of a PLR matrix
 *
 * AUTH: Yang Xinshuo
 *
 * DATE: 20140422
 *
 **********************************************************************/
#include "../include/dplr.h"


/*
 * Make a copy of the m-by-m PLR matrix A, B <- A. B must already be
 * initialized.
 */
void dplrcpy(int m, DPlrMtx *A, DPlrMtx *B)
{
    if (m < 0)
    {
        fprintf(stderr, "%s: invalid matrix size %d\n", __FUNCTION__, m);
        return;
    }
	else if (A->type != PLR_DMTX && A->type != PLR_HMTX)
    { 
        fprintf(stderr, "%s: invalid A type\n", __FUNCTION__);
        return;
    }
    else if (A->size != m)
    {
    	fprintf(stderr, "%s: invalid A size\n", __FUNCTION__);
        return;
    }

    int one = 1, len;

    if (A->type == PLR_DMTX)
    {
    	/*
    	 * Make sure B has appropriate structure
    	 */
    	if (B->type != PLR_DMTX || B->size != m)
    	{
    		dplrclr(B);
    		dplrinid(m, B);
    	}

    	/*
    	 * Do the copy
    	 */
    	len = m*m;
    	dcopy_(&len, A->dd.d, &one, B->dd.d, &one);
    }
    else
    {
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
         * Increase capacity if needed
         */
    	if (B->dh.ne->dlr.capacity < A->dh.ne->dlr.rank)
    		dplrresizelr(B->dh.ne, A->dh.ne->dlr.rank);
        if (B->dh.sw->dlr.capacity < A->dh.sw->dlr.rank)
            dplrresizelr(B->dh.sw, A->dh.sw->dlr.rank);

        /*
         * Do the copy
         */
    	len = A->dh.ne->dlr.rank*m/2;
    	dcopy_(&A->dh.ne->dlr.rank, A->dh.ne->dlr.sig, &one, B->dh.ne->dlr.sig, &one);
    	dcopy_(&len, A->dh.ne->dlr.u, &one, B->dh.ne->dlr.u, &one);
    	dcopy_(&len, A->dh.ne->dlr.v, &one, B->dh.ne->dlr.v, &one);
    	B->dh.ne->dlr.rank = A->dh.ne->dlr.rank;


    	len = A->dh.sw->dlr.rank*m/2;
    	dcopy_(&A->dh.sw->dlr.rank, A->dh.sw->dlr.sig, &one, B->dh.sw->dlr.sig, &one);
    	dcopy_(&len, A->dh.sw->dlr.u, &one, B->dh.sw->dlr.u, &one);
    	dcopy_(&len, A->dh.sw->dlr.v, &one, B->dh.sw->dlr.v, &one);
    	B->dh.sw->dlr.rank = A->dh.sw->dlr.rank;

    	/*
    	 * Copy recursively
    	 */
    	dplrcpy(m/2, A->dh.nw, B->dh.nw);
    	dplrcpy(m/2, A->dh.se, B->dh.se);
    }
}
/***********************************************************************
 *
 * END: dplrcpy.c
 *
 **********************************************************************/
