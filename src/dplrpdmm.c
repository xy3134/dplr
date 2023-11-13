/***********************************************************************
 *
 * NAME: dplrpdmm.c
 *
 * DESC: Perform matrix-matrix operations for PLR matrices with dense
 *		 matrices
 *
 * AUTH: Yang Xinshuo
 *
 * DATE: 20141103
 *
 ***********************************************************************/
#include "../include/dplr.h"

/*
 * Perform one the operations:
 * C := alpha * op(A) * op(B) + beta * C
 * where A is a PLR low rank matrix, B and C are dense matrices, alpha and beta are scalars
 * m: Integer. m specifies the size of PLR matrix A, i.e. A is converted from a m-by-m matrix
 * n: Integer. n specifies the number of columns of the matrix op(B) and the number of columns of the matrix C
 * ldb and ldc are leading dimensions of B and C
 */
static void dplrldmm(char transa, char transb, int m, int n, double *alpha, 
    DPlrMtx *A, double *B, int ldb, double *beta, double *C, int ldc)
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
    
    int one = 1;
    double done = 1.0, dzero = 0.0;
    char TT = 'T', NN = 'N';
    int i, j;

    /*
     * Quick return if possible
     */
    if (m == 0 || n == 0)
        return;

    /*
     * if alpha = 0 or rank of A is 0
     */
    if (*alpha == 0.0 || A->dlr.rank == 0)
    {
        if (*beta == 0.0)
        {
            for (i = 0; i < n; ++i)
                for (j = 0; j < m; ++j)
                    C[i*ldc+j] = 0.0;
        }
        else
        {
            for (i = 0; i < n; ++i)
                dscal_(&m, beta, C+i*ldc, &one);
        }
        return;
    }

    /*
     * Start operations.
     */
    double *temp;
    temp = malloc(A->dlr.rank*n*sizeof(double));

    /*
     * Compute y := U * S * V^T * x
     */
    if (transa == 'N' || transa == 'n')
    {
        if (transb == 'N' || transb == 'n')
        {
            /*
             * temp <- v^T * B
             */
            dgemm_(&TT, &NN, &A->dlr.rank, &n, &m, &done, A->dlr.v, &m, B, &ldb, &dzero, temp, &A->dlr.rank);

            /*
             * temp <- s * temp
             */
            for (i = 0; i < A->dlr.rank; ++i) dscal_(&n, A->dlr.sig+i, temp+i, &A->dlr.rank);

            /*
             * C <- alpha * u * temp + beta * C
             */
            dgemm_(&NN, &NN, &m, &n, &A->dlr.rank, alpha, A->dlr.u, &m, temp, &A->dlr.rank, beta, C, &ldc);

        }
        else
        {
            /*
             * temp <- v^T * B^T
             */
            dgemm_(&TT, &TT, &A->dlr.rank, &n, &m, &done, A->dlr.v, &m, B, &ldb, &dzero, temp, &A->dlr.rank);

            /*
             * temp <- s * temp
             */
            for (i = 0; i < A->dlr.rank; ++i) dscal_(&n, A->dlr.sig+i, temp+i, &A->dlr.rank);

            /*
             * C <- alpha * u * temp + beta * C
             */
            dgemm_(&NN, &NN, &m, &n, &A->dlr.rank, alpha, A->dlr.u, &m, temp, &A->dlr.rank, beta, C, &ldc);   
        }
    }
    else
    {
        if (transb == 'N' || transb == 'n')
        {
            /*
             * temp <- u^T * B
             */
            dgemm_(&TT, &NN, &A->dlr.rank, &n, &m, &done, A->dlr.u, &m, B, &ldb, &dzero, temp, &A->dlr.rank);

            /*
             * temp <- s * temp
             */
            for (i = 0; i < A->dlr.rank; ++i) dscal_(&n, A->dlr.sig+i, temp+i, &A->dlr.rank);

            /*
             * C <- alpha * v * temp + beta * C
             */
            dgemm_(&NN, &NN, &m, &n, &A->dlr.rank, alpha, A->dlr.v, &m, temp, &A->dlr.rank, beta, C, &ldc);
        }
        else
        {
            /*
             * temp <- u^T * B^T
             */
            dgemm_(&TT, &TT, &A->dlr.rank, &n, &m, &done, A->dlr.u, &m, B, &ldb, &dzero, temp, &A->dlr.rank);

            /*
             * temp <- s * temp
             */
            for (i = 0; i < A->dlr.rank; ++i) dscal_(&n, A->dlr.sig+i, temp+i, &A->dlr.rank);

            /*
             * C <- alpha * v * temp + beta * C
             */
            dgemm_(&NN, &NN, &m, &n, &A->dlr.rank, alpha, A->dlr.v, &m, temp, &A->dlr.rank, beta, C, &ldc);   
        }
    }

    /*
     * Free memory
     */
    free(temp);

    /*
     * Success
     */
    return;
}

/*
 * Perform the PLR-matrix operations:
 * C := alpha * A * B + beta * C
 * where B and C are double precision arrays, alpha and beta are scalars, A is a PLR matrix.
 * m: Integer. m specifies the size of PLR matrix A, i.e. A is converted from a m-by-m matrix
 * n: Integer. n specifies the number of columns of the matrix op(B) and the number of columns
 * of the matrix C
 * ldb and ldc are leading dimensions of B and C
 */
static void dplrpdmm(int m, int n, double *alpha, DPlrMtx *A, double *B, 
    int ldb, double *beta, double *C, int ldc)
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
    else if (ldc < m)
    {
        fprintf(stderr, "%s: invalid leading dimensions\n", __FUNCTION__);
        return; 
    }

    char NN = 'N';
    double done = 1.0;


    /*
     * Quick return if possible
     */
    if (m == 0 || n == 0)
        return;


    /*
     * Start operations
     */
    if (A->type == PLR_DMTX)
    {
        /*
         * Dense blocks multiplications
         */
        dgemm_(&NN, &NN, &m, &n, &m, alpha, A->dd.d, &m, B, &ldb, beta, C, &ldc);
    }
    else
    {
        /*
         * Low rank blocks multiplications
         */
        dplrldmm(NN, NN, m/2, n, alpha, A->dh.ne, B+m/2, ldb, beta, C, ldc);
        dplrldmm(NN, NN, m/2, n, alpha, A->dh.sw, B, ldb, beta, C+m/2, ldc);

        /*
         * Recurse on finer levels
         */
        dplrpdmm(m/2, n, alpha, A->dh.nw, B, ldb, &done, C, ldc);
        dplrpdmm(m/2, n, alpha, A->dh.se, B+m/2, ldb, &done, C+m/2, ldc); 
    }  

    /*
     * Success
     */
    return;
}


/*
 * Perform the PLR-matrix operations:
 * C := alpha * A**T * B + beta * C
 * where B and C are double precision arrays, alpha and beta are scalars, A is a PLR matrix.
 * m: Integer. m specifies the size of PLR matrix A, i.e. A is converted from a m-by-m matrix
 * n: Integer. n specifies the number of columns of the matrix op(B) and the number of columns
 * of the matrix C
 * ldb and ldc are leading dimensions of B and C
 */
static void dplrptdmm(int m, int n, double *alpha, DPlrMtx *A, double *B, 
    int ldb, double *beta, double *C, int ldc)
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
    else if (ldc < m)
    {
        fprintf(stderr, "%s: invalid leading dimensions\n", __FUNCTION__);
        return; 
    }

    char NN = 'N', TT = 'T';
    double done = 1.0;

    /*
     * Start operations
     */
    if (A->type == PLR_DMTX)
    {
        /*
         * Dense blocks multiplications
         */
        dgemm_(&TT, &NN, &m, &n, &m, alpha, A->dd.d, &m, B, &ldb, beta, C, &ldc);
    }
    else
    {
        /*
         * Low rank blocks multiplications
         */
        dplrldmm(TT, NN, m/2, n, alpha, A->dh.sw, B+m/2, ldb, beta, C, ldc);
        dplrldmm(TT, NN, m/2, n, alpha, A->dh.ne, B, ldb, beta, C+m/2, ldc);

        /*
         * Recurse on finer levels
         */
        dplrptdmm(m/2, n, alpha, A->dh.nw, B, ldb, &done, C, ldc);
        dplrptdmm(m/2, n, alpha, A->dh.se, B+m/2, ldb, &done, C+m/2, ldc); 
    }  

    /*
     * Success
     */
    return;
}

/*
 * Perform the PLR-matrix operations:
 * C := alpha * A * B**T + beta * C
 * where B and C are double precision arrays, alpha and beta are scalars, A is a PLR matrix.
 * m: Integer. m specifies the size of PLR matrix A, i.e. A is converted from a m-by-m matrix
 * n: Integer. n specifies the number of columns of the matrix op(B) and the number of columns
 * of the matrix C
 * ldb and ldc are leading dimensions of B and C
 */
static void dplrpdtmm(int m, int n, double *alpha, DPlrMtx *A, double *B, 
    int ldb, double *beta, double *C, int ldc)
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
    else if (ldc < m)
    {
        fprintf(stderr, "%s: invalid leading dimensions\n", __FUNCTION__);
        return; 
    }

    char NN = 'N', TT = 'T';
    double done = 1.0;

    /*
     * Start operations
     */
    if (A->type == PLR_DMTX)
    {
        /*
         * Dense blocks multiplications
         */
        dgemm_(&NN, &TT, &m, &n, &m, alpha, A->dd.d, &m, B, &ldb, beta, C, &ldc);
    }
    else
    {
        /*
         * Low rank blocks multiplications
         */
        dplrldmm(NN, TT, m/2, n, alpha, A->dh.ne, B+m/2*ldb, ldb, beta, C, ldc);
        dplrldmm(NN, TT, m/2, n, alpha, A->dh.sw, B, ldb, beta, C+m/2, ldc);

        /*
         * Recurse on finer levels
         */
        dplrpdtmm(m/2, n, alpha, A->dh.nw, B, ldb, &done, C, ldc);
        dplrpdtmm(m/2, n, alpha, A->dh.se, B+m/2*ldb, ldb, &done, C+m/2, ldc); 
    }  

    /*
     * Success
     */
    return;
}

/*
 * Perform the PLR-matrix operations:
 * C := alpha * A**T * B**T + beta * C
 * where B and C are double precision arrays, alpha and beta are scalars, A is a PLR matrix.
 * m: Integer. m specifies the size of PLR matrix A, i.e. A is converted from a m-by-m matrix
 * n: Integer. n specifies the number of columns of the matrix op(B) and the number of columns
 * of the matrix C
 * ldb and ldc are leading dimensions of B and C
 */
static void dplrptdtmm(int m, int n, double *alpha, DPlrMtx *A, double *B, 
    int ldb, double *beta, double *C, int ldc)
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
    else if (ldc < m)
    {
        fprintf(stderr, "%s: invalid leading dimensions\n", __FUNCTION__);
        return; 
    }

    char TT = 'T';
    double done = 1.0;

    /*
     * Start operations
     */
    if (A->type == PLR_DMTX)
    {
        /*
         * Dense blocks multiplications
         */
        dgemm_(&TT, &TT, &m, &n, &m, alpha, A->dd.d, &m, B, &ldb, beta, C, &ldc);
    }
    else
    {
        /*
         * Low rank blocks multiplications
         */
        dplrldmm(TT, TT, m/2, n, alpha, A->dh.sw, B+m/2*ldb, ldb, beta, C, ldc);
        dplrldmm(TT, TT, m/2, n, alpha, A->dh.ne, B, ldb, beta, C+m/2, ldc);

        /*
         * Recurse on finer levels
         */
        dplrptdtmm(m/2, n, alpha, A->dh.nw, B, ldb, &done, C, ldc);
        dplrptdtmm(m/2, n, alpha, A->dh.se, B+m/2*ldb, ldb, &done, C+m/2, ldc); 
    }  

    /*
     * Success
     */
    return;
}


/*
 * Perform one of the PLR-matrix operations:
 * C := alpha*op(A)*op(B) + beta*C
 * where B and C are double precision arrays, alpha and beta are scalars, A is a PLR matrix.
 * m: Integer. m specifies the size of PLR matrix A, i.e. A is converted from a m-by-m matrix
 * n: Integer. n specifies the number of columns of the matrix op(B) and the number of columns of the matrix C
 * ldb and ldc are leading dimensions of B and C
 */
void dplrgepdmm(char transa, char transb, int m, int n, double *alpha, 
	DPlrMtx *A, double *B, int ldb, double *beta, double *C, int ldc)
{

	/*
     * No need to check inputs, since subroutines will check.
     * Start operations
     */
    if (transa == 'N' || transa == 'n')
    {
        if (transb == 'N' || transb == 'n')
        {
            dplrpdmm(m, n, alpha, A, B, ldb, beta, C, ldc);
        }
        else
        {
            dplrpdtmm(m, n, alpha, A, B, ldb, beta, C, ldc);
        }
    }
    else
    {

        if (transb == 'N' || transb == 'n')
        {
            dplrptdmm(m, n, alpha, A, B, ldb, beta, C, ldc);
        }
        else
        {
            dplrptdtmm(m, n, alpha, A, B, ldb, beta, C, ldc);
        }
    }

    /*
     * Success
     */
    return;
}

/***********************************************************************
 *
 * END: dplrpdmm.c
 *
 ***********************************************************************/

