/***********************************************************************
 *
 * NAME: dplrprk.c
 *
 * DESC: Perform a rank-k update to a PLR matrices
 *
 * AUTH: Yang Xinshuo
 *
 * DATE: 20141103
 *
 ***********************************************************************/
#include "../include/dplr.h"


/*
 * Performs a rank-k update to a PLR matrix A:
 * A <- A + alpha * u * s * v'
 * where  alpha  is  a scalar, u and v are m-by-k dense matrices, s is a k-by-k diagonal matrix
 * supplied as a length-k vector and A is an m-by-m PLR matrix. eps gives absolute error threshold.
 */
void dplrgeprk(int m, int k, double *alpha, double *u, int ldu, double *v, 
			int ldv, double *s, int incs, DPlrMtx *A, double eps)
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
    else if (ldu < m || ldv < m)
    {
        fprintf(stderr, "%s: invalid leading dimensions\n", __FUNCTION__);
        return;  
    }

    /*
     * If alpha = 0 or k = 0
     */
    if (*alpha == 0.0 || k == 0)
        return;


    char NN = 'N', TT = 'T';
    int one = 1;
    double done = 1.0;
    int i, len, r1, r2;
    double *temp, f1, f2, fff, eps2;

    /*
     * Start operations
     */
    if (A->type == PLR_DMTX)
    {
        temp = malloc(m*k*sizeof(double));

        for (i = 0; i < k; ++i)
        {
            dcopy_(&m, u+i*ldu, &one, temp+i*m, &one);
            dscal_(&m, s+i*incs ,temp+i*m, &one);
        }
        dgemm_(&NN, &TT, &m, &m ,&k, alpha, temp, &m, v, &ldv, &done, A->dd.d, &m);

        free(temp);
    }
    else
    {   
        /*
         * Determine if next level is the last level of decomposition
         * if so allocate error to the two low rank blocks
         * otherwise allocate error to all 4 blocks
         */
        if (A->dh.nw->type == PLR_DMTX)
        {
            eps2 = eps/2.0;
        }
        else
        {
            eps2 = eps/4.0;
        }


        r1 = A->dh.ne->dlr.rank;
        r2 = A->dh.sw->dlr.rank;
        len = m/2;
        fff = fabs(*alpha);

        /*
         * Increase capacity of A->dh.ne if necessary
         */
        if (A->dh.ne->dlr.capacity < r1+k)
            dplrresizelr(A->dh.ne, r1+k);
        
        /*
         * Copy u and v to A->dh.ne
         */
        for (i = 0; i < k; ++i)
        {
            dcopy_(&len, u+i*ldu, &one, A->dh.ne->dlr.u+(i+r1)*len, &one);
            dcopy_(&len, v+i*ldv+len, &one, A->dh.ne->dlr.v+(i+r1)*len, &one);

            /*
             * Normalize vectors
             */
            f1 = dnrm2_(&len, A->dh.ne->dlr.u+(i+r1)*len, &one);
            f2 = dnrm2_(&len, A->dh.ne->dlr.v+(i+r1)*len, &one);

            if (fff * s[i*incs] * f1 * f2 < eps2)
            {
                *(A->dh.ne->dlr.sig+r1+i) = 0.0;   
            }
            else
            {
                *(A->dh.ne->dlr.sig+r1+i) = fff * s[i*incs] * f1 * f2;

                if (*alpha > 0.0)
                {
                    f1 = 1.0/f1;
                }
                else
                {
                    f1 = -1.0/f1;
                }

                f2 = 1.0/f2;
                dscal_(&len, &f1, A->dh.ne->dlr.u+(r1+i)*len, &one);
                dscal_(&len, &f2, A->dh.ne->dlr.v+(r1+i)*len, &one);
            }
        }

        /*
         * Adjust ranks
         */
        A->dh.ne->dlr.rank += k;

        /*
         * Increase capacity of A->dh.sw if necessary
         */
        if (A->dh.sw->dlr.capacity < r2+k)
            dplrresizelr(A->dh.sw, r2+k);
        
        /*
         * Copy u and v to A->dh.sw
         */
        for (i = 0; i < k; ++i)
        {
            dcopy_(&len, u+i*ldu+len, &one, A->dh.sw->dlr.u+(i+r2)*len, &one);
            dcopy_(&len, v+i*ldv, &one, A->dh.sw->dlr.v+(i+r2)*len, &one);

            /*
             * Normalize vectors
             */

            f1 = dnrm2_(&len, A->dh.sw->dlr.u+(i+r2)*len, &one);
            f2 = dnrm2_(&len, A->dh.sw->dlr.v+(i+r2)*len, &one);

            if (fff * s[i*incs] * f1 * f2 < eps2)
            {
                *(A->dh.sw->dlr.sig+r2+i) = 0.0;   
            }
            else
            {
                *(A->dh.sw->dlr.sig+r2+i) = fff * s[i*incs] * f1 * f2;

                if (*alpha > 0.0)
                {
                    f1 = 1.0/f1;
                }
                else
                {
                    f1 = -1.0/f1;
                }
                
                f2 = 1.0/f2;
                dscal_(&len, &f1, A->dh.sw->dlr.u+(r2+i)*len, &one);
                dscal_(&len, &f2, A->dh.sw->dlr.v+(r2+i)*len, &one);
            }
        }

        /*
         * Adjust ranks
         */
        A->dh.sw->dlr.rank += k;

        /*
         * Update diagonal blocks recursively
         */
        dplrgeprk(len, k, alpha, u, ldu, v, ldv, s, incs, A->dh.nw, eps2);
        dplrgeprk(len, k, alpha, u+len, ldu, v+len, ldv, s, incs, A->dh.se, eps2);
    }
}


 /***********************************************************************
 *
 * END: dplrprk.c
 *
 ***********************************************************************/


