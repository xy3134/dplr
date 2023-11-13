/***********************************************************************
 *
 * NAME: dplrinv.c
 *
 * DESC: Compute inverse of PLR matrices.
 *
 * AUTH: Xinshuo Yang
 *
 * DATE: 20150322
 *
 ***********************************************************************/

#include "../include/dplr.h"

/*
 * Compute inverse of a m-by-m dense matrix A, i.e.
 * A <- inv(A)
 * lda specifies the leading dimension of A
 */
static void dplrdinv(int m, double *A, int lda, int *info)
{
    int *ipiv, lwork;
    double workq, *work;
    
    if (lda < m)
    {
        fprintf(stderr, "%s: invalid leading dimension\n", __FUNCTION__);
    }

    *info = -1;

    /*
     * Compute LU factorization of A
     */
    ipiv = malloc(m*sizeof(int));
    dgetrf_(&m, &m, A, &lda, ipiv, info);
    if (*info != 0)
    {
        fprintf(stderr, "%s: dgetrf() failed: %d\n", __FUNCTION__, *info);
        free(ipiv);
        return;   
    }

    /*
     * dgetri workspace query
     */
    lwork = -1;
    dgetri_(&m, A, &lda, ipiv, &workq, &lwork, info);
    if (*info != 0)
    {
        fprintf(stderr, "%s: dgetri() workspace query failed: %d\n",
                __FUNCTION__, *info);
        free(ipiv);
        return;
    }

    lwork = (int)(workq);
    work = malloc(lwork*sizeof(double));

    /*
     * Compute inverse from LU factorization
     */
    dgetri_(&m, A, &lda, ipiv, work, &lwork, info);
    if (*info != 0)
    {
        fprintf(stderr, "%s: dgetri() failed: %d\n", 
            __FUNCTION__, *info);
        free(work);
        free(ipiv);
        return;   
    }

    /*
     * Free memory
     */
    free(ipiv);
    free(work);

    /*
     * Success
     */
    *info = 0;
    return;
}


/*
 * Compute inverse of a m-by-m PLR matrix A, i.e.
 * B <- inv(A)
 * B must already be initialized. 
 * eps specifies absolute error threshold.
 */
void dplrgeinv(int m, DPlrMtx *A, DPlrMtx *B, double eps, int *info)
{   
    char NN = 'N', TT = 'T';
    int one = 1;
    double dnone = -1.0, dzero = 0.0, done = 1.0;
    int i, j, len, r1, r2, r, ll;
    double eps2, f1, f2, *u1, *d1, *v1, *u2, *d2, *v2, *temp;


    if (m < 0)
    {
        fprintf(stderr, "%s: invalid matrix size: %d\n", __FUNCTION__, m);
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


    *info = -1;

    /*
     * Start operations.
     */
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
         * Copy A to B
         */
        len = m*m;
        dcopy_(&len, A->dd.d, &one, B->dd.d, &one);
        
        /*
         * Compute inverse
         */
        dplrdinv(m, B->dd.d, m, info);
        if (*info != 0)
        {
            fprintf(stderr, "%s: failed to invert a dense matix: %d\n",
             __FUNCTION__, *info);
            return;
        }
    }
    else
    {
        if (A->dh.nw->type == PLR_DMTX)
        {
            /*
             * If the next level is the last level of decomposition, we 
             * will allocate error to each of the two low rank blocks. 
             * The diagonal blocks are dense and therefore they do not 
             * have any approximation error.
             */
            eps2 = eps/2.0;
        }
        else
        {
            /*
             * If the next level is not the last level of decomposition, we
             * will allocate error to all the 4 blocks.
             */
            eps2 = eps/4.0;
        }

        /*
         * Make sure B has appropriate structure
         */
        if (B->type != PLR_HMTX || B->size != m)
        {
            dplrclr(B);
            dplrinih(m, B);
        }

        /*
         * Compute inverse of diagonal blocks recursively
         */
        dplrgeinv(m/2, A->dh.nw, B->dh.nw, eps2, info);
        if (*info != 0)
        {
            fprintf(stderr, "%s: failed to invert nw block: %d\n",
             __FUNCTION__, *info);
            return;
        }

        dplrgeinv(m/2, A->dh.se, B->dh.se, eps2, info);
        if (*info != 0)
        {
            fprintf(stderr, "%s: failed to invert se block: %d\n",
             __FUNCTION__, *info);
            return;
        }

        /*
         * Make sure low rank blocks have proper structure
         */
        if (B->dh.ne->type != PLR_LRMTX || B->dh.ne->size != m/2)
        {
            dplrclr(B->dh.ne);
            dplrinilr(m/2, B->dh.ne, A->dh.ne->dlr.rank);
        }
        if (B->dh.sw->type != PLR_LRMTX || B->dh.sw->size != m/2)
        {
            dplrclr(B->dh.sw);
            dplrinilr(m/2, B->dh.sw, A->dh.sw->dlr.rank);
        }

        /*
         * Increase capacity if needed
         */
        if (B->dh.ne->dlr.capacity < A->dh.ne->dlr.rank)    
            dplrresizelr(B->dh.ne, A->dh.ne->dlr.rank);
        if (B->dh.sw->dlr.capacity < A->dh.sw->dlr.rank)    
            dplrresizelr(B->dh.sw, A->dh.sw->dlr.rank);


        /*
         * If ne and sw blocks both have rank 0, we are done
         * Otherwise perform a low rank update
         */
        if (A->dh.ne->dlr.rank == 0 && A->dh.sw->dlr.rank == 0)
        {
            ;
        }
        else        
        {
            len = m/2;
            r1 = A->dh.ne->dlr.rank;
            r2 = A->dh.sw->dlr.rank;
            r = r1+r2;

            /*
             * Initialize buffers to store low rank updates
             */
            u1 = malloc(len*r1*sizeof(double));
            d1 = malloc(r1*sizeof(double));
            v1 = malloc(len*r1*sizeof(double));
            u2 = malloc(len*r2*sizeof(double));
            d2 = malloc(r2*sizeof(double));
            v2 = malloc(len*r2*sizeof(double));

            /*
             * u1 <- (A->dh.nw)^-1 * A->dh.ne->dlr.u
             * u2 <- (A->dh.se)^-1 * A->dh.sw->dlr.u
             */
            dplrgepdmm(NN, NN, len, r1, &done, B->dh.nw, A->dh.ne->dlr.u, len, &dzero, u1, len);
            dplrgepdmm(NN, NN, len, r2, &done, B->dh.se, A->dh.sw->dlr.u, len, &dzero, u2, len);


            /*
             * Compute:
             * B->dh.ne->dlr.v := (A->dh.se')^-1 * A->dh.ne->dlr.v
             * B->dh.sw->dlr.v := (A->dh.nw')^-1 * A->dh.sw->dlr.v
             */
            dplrgepdmm(TT, NN, len, r1, &done, B->dh.se, A->dh.ne->dlr.v, len, &dzero, B->dh.ne->dlr.v, len);
            dplrgepdmm(TT, NN, len, r2, &done, B->dh.nw, A->dh.sw->dlr.v, len, &dzero, B->dh.sw->dlr.v, len);
            
            /*
             * Copy B->dh.ne->dlr.v to v1, copy B->dh.sw->dlr.v to v2
             */
            ll = len*r1;
            dcopy_(&ll, B->dh.ne->dlr.v, &one, v1, &one);
            ll = len*r2;
            dcopy_(&ll, B->dh.sw->dlr.v, &one, v2, &one);

            /*
             * Generate the (r1+r2)-by-(r1+r2) matrix X, 
             * compute inverse
             */
            double *X;
            X = malloc(r*r*sizeof(double));

            /*
             * Construct diagonal blocks of X 
             */
            for (i = 0; i < r1; ++i)
            {
                for (j = 0; j < r1; ++j)
                {
                    if (i == j)
                        *(X+j*r+i) = 1.0/(*(A->dh.ne->dlr.sig+i));
                    else
                        *(X+j*r+i) = 0.0;
                }
            }
            for (i = 0; i < r2; ++i)
            {
                for (j = 0; j < r2; ++j)
                {
                    if (i == j)
                        *(X+r1*r+j*r+i+r1) = 1.0/(*(A->dh.sw->dlr.sig+i));
                    else
                        *(X+r1*r+j*r+i+r1) = 0.0;
                }
            }

            /*
             * Cunstruct off-diagonal blocks of X
             */
            dgemm_(&TT, &NN, &r1, &r2, &len, &done, A->dh.ne->dlr.v, &len, u2, &len,
                            &dzero, X+r1*r, &r);
            dgemm_(&TT, &NN, &r2, &r1, &len, &done, A->dh.sw->dlr.v, &len, u1, &len,
                            &dzero, X+r1, &r);

            /*
             * Compute inverse of X
             */
            dplrdinv(r, X, r, info);
            if (*info != 0)
            {
                fprintf(stderr, "%s: failed to invert the dense matrix\
                 of Woodbury Formula: %d\n", __FUNCTION__, *info);
                return;
            }


            /*
             * Compute B->dh.ne->dlr.u and B->dh.sw->dlr.u
             */
            dgemm_(&NN, &NN, &len, &r1, &r1, &dnone, u1, &len, X, &r, &dzero, B->dh.ne->dlr.u, &len);
            dgemm_(&NN, &NN, &len, &r2, &r2, &dnone, u2, &len, X+r1*r+r1, &r, &dzero, B->dh.sw->dlr.u, &len);

            /*
             * Normalize off-diagonal blocks, adjust s-values and rank accordingly
             */
            for (i = 0; i < r1; ++i)
            {
                f1 = dnrm2_(&len, B->dh.ne->dlr.u+i*len, &one);
                f2 = dnrm2_(&len, B->dh.ne->dlr.v+i*len, &one);
                if (f1*f2 < eps2)
                {
                    *(B->dh.ne->dlr.sig+i) = 0.0;
                }
                else
                {
                    *(B->dh.ne->dlr.sig+i) = f1*f2;
                    f1 = 1.0/f1;
                    f2 = 1.0/f2;
                    dscal_(&len, &f1, B->dh.ne->dlr.u+i*len, &one);
                    dscal_(&len, &f2, B->dh.ne->dlr.v+i*len, &one);
                }
            }
            B->dh.ne->dlr.rank = r1;

            for (i = 0; i < r2; ++i)
            {
                f1 = dnrm2_(&len, B->dh.sw->dlr.u+i*len, &one);
                f2 = dnrm2_(&len, B->dh.sw->dlr.v+i*len, &one);
                if (f1*f2 < eps2)
                {
                    *(B->dh.sw->dlr.sig+i) = 0.0;
                }
                else
                {
                    *(B->dh.sw->dlr.sig+i) = f1*f2;
                    f1 = 1.0/f1;
                    f2 = 1.0/f2;
                    dscal_(&len, &f1, B->dh.sw->dlr.u+i*len, &one);
                    dscal_(&len, &f2, B->dh.sw->dlr.v+i*len, &one);
                }
            }
            B->dh.sw->dlr.rank = r2;

            /*
             * Compute
             * u1 := u2 * x2
             * u2 := u1 * x3
             */
            ll = len*r1;
            temp = malloc(ll*sizeof(double));
            dcopy_(&ll, u1, &one, temp, &one);
            dgemm_(&NN, &NN, &len, &r1, &r2, &dnone, u2, &len, X+r1, &r, &dzero, u1, &len);
            dgemm_(&NN, &NN, &len, &r2, &r1, &dnone, temp, &len, X+r1*r, &r, &dzero, u2, &len);

            /*
             * Normalize u2, v2, adjust s-values and rank
             */
            for (i = 0; i < r2; ++i)
            {
                f1 = dnrm2_(&len, u2+i*len, &one);
                f2 = dnrm2_(&len, v2+i*len, &one);
                if (f1*f2 < eps2)
                {
                    *(d2+i) = 0.0;
                }
                else
                {
                    *(d2+i) = f1*f2;
                    f1 = 1.0/f1;
                    f2 = 1.0/f2;
                    dscal_(&len, &f1, u2+i*len, &one);
                    dscal_(&len, &f2, v2+i*len, &one);
                }
            }

            /*
             * Normalize u1, v1, adjust s-values and rank
             */
            for (i = 0; i < r1; ++i)
            {
                f1 = dnrm2_(&len, u1+i*len, &one);
                f2 = dnrm2_(&len, v1+i*len, &one);

                if (f1*f2 < eps2)
                {
                    *(d1+i) = 0.0;
                }
                else
                {
                    *(d1+i) = f1*f2;
                    f1 = 1.0/f1;
                    f2 = 1.0/f2;
                    dscal_(&len, &f1, u1+i*len, &one);
                    dscal_(&len, &f2, v1+i*len, &one);
                }
            }

            /*
             * Add low rank corrections to B->dh.nw and B->dh.se
             */
            dplrgeprk(len, r2, &done, u2, len, v2, len, d2, 1, B->dh.nw, eps2);
            dplrgeprk(len, r1, &done, u1, len, v1, len, d1, 1, B->dh.se, eps2);

            /*
             * Free memory
             */
            free(X);
            free(u1);
            free(d1);
            free(v1);
            free(u2);
            free(d2);
            free(v2);
            free(temp);
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
 * END: dplrinv.c
 *
 ***********************************************************************/

