/***********************************************************************
 *
 * NAME: dplrnz.c
 *
 * DESC: Compute the non-zero entries of a PLR matrix
 *
 * AUTH: Yang Xinshuo
 *
 * DATE: 20160411
 *
 ***********************************************************************/
#include "../include/dplr.h"

/*
 * Compute the total non-zero entries of a m-by-m PLR matrix
 */
long int dplrnz(int m, DPlrMtx *A)
{
	long int nz;

    if (m < 0)
    {
        fprintf(stderr, "%s: invalid matrix size %d\n", __FUNCTION__, m);
        return -1;   
    }
    else if (A->size != m)
    {
        fprintf(stderr, "%s: invalid A size\n", __FUNCTION__);
        return -1;
    }
    else if (A->type != PLR_DMTX && A->type != PLR_HMTX)
    {
        fprintf(stderr, "%s: invalid A type\n", __FUNCTION__);
        return -1;   
    }

    /*
     * Start operations
     */
    if (A->type == PLR_DMTX)
    {
    	/*
    	 * Non-zero entries are the m-by-m dense matrix
    	 */
    	nz = m*m;
    }
    else
    {
    	/*
    	 * Non-zero entries of low rank matrix
    	 * Ignore s-values
    	 */
    	nz = m*(A->dh.ne->dlr.rank+A->dh.sw->dlr.rank);

    	/*
    	 * Recursion
    	 */
    	nz += dplrnz(m/2, A->dh.nw);
    	nz += dplrnz(m/2, A->dh.se);
    }

    return nz;
}
/***********************************************************************
 *
 * END: dplrnz.c
 *
 ***********************************************************************/
