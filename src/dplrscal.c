/***********************************************************************
 *
 * NAME: dplrscal.c
 *
 * DESC: Scale a PLR matrix by a scalar.
 *
 * AUTH: Yang Xinshuo
 *
 * DATE: 20140422
 *
 **********************************************************************/
#include "../include/dplr.h"
 
/*
 * Scale the m-by-m PLR matrix A by a scalar, i.e. 
 * A <- alpha * A.
 */
void dplrscal(int m, double *alpha, DPlrMtx *A)
{	
    if (m < 0)
    { 
        fprintf(stderr, "%s: invalid matrix size\n", __FUNCTION__);
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
    

    int one = 1;
    int len;

    if (A->type == PLR_DMTX)
    {	
    	len = m*m;
    	dscal_(&len, alpha, A->dd.d, &one);
    }
    else
    {
    	/*
    	 * Scale low rank parts. If alpha>0, only scale s-values, 
    	 * othewise scale s-values by -alpha, and negative u vectors.
    	 * This matters because we want s-values to be positive.
    	 */
    	if (*alpha >= 0.0)
    	{
    		dscal_(&A->dh.ne->dlr.rank, alpha, A->dh.ne->dlr.sig, &one);
    		dscal_(&A->dh.sw->dlr.rank, alpha, A->dh.sw->dlr.sig, &one);
    	}
    	else
    	{
    		double fff = -(*alpha);
    		double dnone = -1.0;
    		
    		len = A->dh.ne->dlr.rank*m/2;
    		dscal_(&A->dh.ne->dlr.rank, &fff, A->dh.ne->dlr.sig, &one);
    		dscal_(&len, &dnone, A->dh.ne->dlr.u, &one);

    		len = A->dh.sw->dlr.rank*m/2;
    		dscal_(&A->dh.sw->dlr.rank, &fff, A->dh.sw->dlr.sig, &one);
    		dscal_(&len, &dnone, A->dh.sw->dlr.u, &one);
    	}

    	dplrscal(m/2, alpha, A->dh.nw);
    	dplrscal(m/2, alpha, A->dh.se);
    }

}

/***********************************************************************
 *
 * END: dplrscal.c
 *
 **********************************************************************/

