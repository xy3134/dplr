/***********************************************************************
 *
 * NAME: dplrpmv.c
 *
 * DESC: Perform matrix-vector operations for PLR matrices.
 *
 * AUTH: Yang Xinshuo
 *
 * DATE: 20141103
 *
 ***********************************************************************/
#include "../include/dplr.h"

/*
 * Perform one of the matrix-vector operations:
 * y := alpha * A * x + beta * y or y := alpha * A' * x + beta * y
 * where x and y are vectors and A is a PLR low-rank matrix.
 */
static void dplrlrmv(char trans, int m, double *alpha, DPlrMtx *A, 
    double *x, int incx, double *beta, double *y, int incy)
{
    if (m < 0)
    {
        fprintf(stderr, "%s: invalid matrix size %d\n", __FUNCTION__, m);
        return;
    }
	else if (A->type != PLR_LRMTX)
    {
        fprintf(stderr, "%s: invalid A type\n", __FUNCTION__);
        return;
    }
	else if (A->size != m)
    {
        fprintf(stderr, "%s: invalid A size\n", __FUNCTION__);
        return;
    }

    char NN = 'N', TT = 'T';
    int one = 1;
    double done = 1.0, dzero = 0.0;
    int i;

    /*
     * Quick return if possible
     */
    if (A->dlr.rank == 0 || *alpha == 0.0)
    {
        if (*beta == 0.0)
        {
            for (i = 0; i < m; ++i)
                y[i*incy] = 0.0;
        }
        else
        {
            dscal_(&m, beta, y, &incy);
        }

        return;
    }

    
    double *temp;
    temp = malloc(A->dlr.rank*sizeof(double));

    if (trans == 'N' || trans == 'n')
    {
    	/*
         * temp <- v' * x
         */
        dgemv_(&TT, &m, &A->dlr.rank, &done, A->dlr.v, &m, x, &incx, &dzero, temp, &one);

        /*
         * temp <- s * temp
         */
        for (i = 0; i < A->dlr.rank; ++i)   temp[i] *= A->dlr.sig[i];

        /*
         * y <- alpha * u * temp
         */
        dgemv_(&NN, &m, &A->dlr.rank, alpha, A->dlr.u, &m, temp, &one, beta, y, &incy);

    }
    else
    {
    	/*
         * temp <- u' * x
         */
        dgemv_(&TT, &m, &A->dlr.rank, &done, A->dlr.u, &m, x, &incx, &dzero, temp, &one);

        /*
         * temp <- s * z
         */
        for (i = 0; i < A->dlr.rank; ++i)   temp[i] *= A->dlr.sig[i];

        /*
         * y <- alpha * v * temp
         */
        dgemv_(&NN, &m, &A->dlr.rank, alpha, A->dlr.v, &m, temp, &one, beta, y, &incy);
    }

    /*
     * Free memory
     */
    free(temp);
}


/*
 * Perform the matrix-vector operations:
 * y := alpha * A * x + beta * y
 * where x and y are vectors, alpha and beta are scalars, A is a m-by-m PLR matrix.
 */
static void dplrpmv(int m, double *alpha, DPlrMtx *A, double *x, 
    int incx, double *beta, double *y, int incy)
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

    double done = 1.0;
    char NN = 'N';
    int i;


    /*
     * If alpha = 0
     */
    if (*alpha == 0.0)
    {
        if (*beta == 0.0)
        {
            for (i = 0; i < m; ++i)
                y[i*incy] = 0.0;
        }
        else
        {
            dscal_(&m, beta, y, &incy);
        }
    }


    /*
     * Start operations
     */
    if (A->type == PLR_DMTX)
    {
         dgemv_(&NN, &m, &m, alpha, A->dd.d, &m, x, &incx, beta, y, &incy);
    }
    else
    {   
        /*
         * Low rank blocks multiply x
         */
        dplrlrmv(NN, m/2, alpha, A->dh.ne, x+m/2*incx, incx, beta, y, incy);
        dplrlrmv(NN, m/2, alpha, A->dh.sw, x, incx, beta, y+m/2*incy, incy);
        
        /*
         * Do the computation recursively
         */
        dplrpmv(m/2, alpha, A->dh.nw, x, incx, &done, y, incy);
        dplrpmv(m/2, alpha, A->dh.se, x+m/2*incx, incx, &done, y+m/2*incy, incy);
    }

    /* 
     * Success
     */
    return;
}


/*
 * Perform the matrix-vector operations:
 * y := alpha * A' * x + beta * y
 * where x and y are vectors, alpha and beta are scalars, A is a m-by-m PLR matrix.
 */
static void dplrptmv(int m, double *alpha, DPlrMtx *A, double *x, 
    int incx, double *beta, double *y, int incy)
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

    double done = 1.0;
    char TT = 'T';
    int i;


    /*
     * If alpha = 0
     */
    if (*alpha == 0.0)
    {
        if (*beta == 0.0)
        {
            for (i = 0; i < m; ++i)
                y[i*incy] = 0.0;
        }
        else
        {
            dscal_(&m, beta, y, &incy);
        }
    }

    /*
     * Start operations
     */
    if (A->type == PLR_DMTX)
    {
        dgemv_(&TT, &m, &m, alpha, A->dd.d, &m, x, &incx, beta, y, &incy);
    }
    else
    {
        /*
         * Low rank blocks multiply x
         */
        dplrlrmv(TT, m/2, alpha, A->dh.sw, x+m/2*incx, incx, beta, y, incy);
        dplrlrmv(TT, m/2, alpha, A->dh.ne, x, incx, beta, y+m/2*incy, incy);
        
        /*
         * Recurse on finer levels
         */
        dplrptmv(m/2, alpha, A->dh.nw, x, incx, &done, y, incy);
        dplrptmv(m/2, alpha, A->dh.se, x+m/2*incx, incx, &done, y+m/2*incy, incy);
    }

    /*
     * Success
     */
    return;
}


/*
 * Perform one of the matrix-vector operations:
 * y := alpha * A * x + beta * y or y := alpha * A' * x + beta * y
 * where x and y are vectors, alpha and beta are scalars, A is a m-by-m PLR matrix.
 */
void dplrgepmv(char trans, int m, double *alpha, DPlrMtx *A, double *x, 
    int incx, double *beta, double *y, int incy)
{
    /*
     * No need to chekc inputs since dplrpmv and dplrptmv will check
     * Start operations
     */
    if (trans == 'N' || trans == 'n')
    {
        dplrpmv(m, alpha, A, x, incx, beta, y, incy);
    }
    else
    {
       dplrptmv(m, alpha, A, x, incx, beta, y, incy);
    }

    /*
     * Success
     */
    return;
}

/***********************************************************************
 *
 * END: dplrpmv.c
 *
 ***********************************************************************/

