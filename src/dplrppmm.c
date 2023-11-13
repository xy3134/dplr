/***********************************************************************
 *
 * NAME: dplrppmm.c
 *
 * DESC: Perform matrix-matrix operations for PLR matrices with PLR
 *		 matrices
 *
 * AUTH: Yang Xinshuo
 *
 * DATE: 20141103
 *
 ***********************************************************************/
#include "../include/dplr.h"

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

/*
 * Performs the following operation:
 * u3 * s3 * v3' <- alpha * u1 * s1 * v3' * u2 * s2 * v2' + beta * u3 * s3 * v3
 * where ui, si, vi is low rank representation of matrix, u_i and v_i are m-by-ri, 
 * si is a length-ri vector, for i = 1, 2, 3.
 * ldui, ldvi specify the leading dimensions of ui, vi
 * incsi specifies the increment between elements of si.
 */
static void dplrgellmm(int m, int r1, int r2, int *r3, double *alpha, double *u1, 
    double *s1, double *v1, double *u2, double *s2, double *v2, 
    double *beta, double *u3, double *s3, double *v3, double eps)
{
    char NN = 'N', TT = 'T';
    int one = 1;
    double dnone = -1.0, dzero = 0.0, done = 1.0;
    int i, len;
    double fff, *temp;

    if (m < 0)
    {
        fprintf(stderr, "%s: invalid matrix size %d\n", __FUNCTION__, m);
        return;
    }

    /*
     * Quick return if possible
     */
    if (m == 0 || r1 == 0 || r2 == 0)
        return;


    /*
     * If alpha = 0
     */
    if (*alpha == 0.0)
    {
        if (*beta == 0.0)
        {
            /*
             * Set r3 to 0
             */
            *r3 = 0;
        }
        else
        {   
            /*
             * Scale C by beta
             */
            if (*beta >= 0.0)
                dscal_(r3, beta, s3, &one);
            else
            {
                fff = -(*beta);
                dscal_(r3, &fff, s3, &one);
                for (i = 0; i < *r3; ++i)
                    dscal_(&m, &dnone, u3+i*m, &one);
            }
        }

        return;
    }


    /*
     * Start operations
     */
    if (r1 < r2)
    {
        /*
         * Copy u1 to u3
         */
        len = m*r1;
        dcopy_(&len, u1, &one, u3+m*(*r3), &one);

        temp = malloc(r1*r2*sizeof(double));
        /*
         * Compute: i) temp := u2' * v1
         *         ii) temp := s2 * temp
         *        iii) v3 := alpha * v2 * temp
         */
        dgemm_(&TT, &NN, &r2, &r1, &m, &done, u2, &m, v1, &m, &dzero, temp, &r2);
        for (i = 0; i < r2; ++i)    dscal_(&r1, s2+i, temp+i, &r2);
        dgemm_(&NN, &NN, &m, &r1, &r2, alpha, v2, &m, temp, &r2, &dzero, v3+*r3*m, &m);

        /*
         * Normalize columns of v3, adjust s3 accordingly
         */
        for (i = 0; i < r1; ++i)
        {
            fff = dnrm2_(&m, v3+(i+*r3)*m, &one);
            if (*(s1+i)*fff < eps)
            {
                *(s3+i+*r3) = 0.0;
            }
            else
            {
                *(s3+i+*r3) = *(s1+i)*fff;
                fff = 1.0/fff;
                dscal_(&m, &fff, v3+(i+*r3)*m, &one);
            }
        }

        /*
         * Adjust rank
         */
        *r3 += r1;

        /*
         * Free memory
         */
        free(temp);
    }
    else
    {
        /*
         * Concatenate v2 to v3
         */
        len = m*r2;
        dcopy_(&len, v2, &one, v3+(*r3)*m, &one);

        temp = malloc(r1*r2*sizeof(double));

        /*
         * Compute: i) temp := v1' * u2
         *         ii) temp := s1 * temp
         *        iii) u3 := alpha * u1 * temp
         */
        dgemm_(&TT, &NN, &r1, &r2, &m, &done, v1, &m, u2, &m, &dzero, temp, &r1);
        for (i = 0; i < r1; ++i)    dscal_(&r2, s1+i, temp+i, &r1);
        dgemm_(&NN, &NN, &m, &r2, &r1, alpha, u1, &m, temp, &r1, &dzero, u3+*r3*m, &m);

        /*
         * Normalize columns of u3, adjust s3 accordingly
         */
        for (i = 0; i < r2; ++i)
        {
            fff = dnrm2_(&m, u3+(i+*r3)*m, &one);
            if (*(s2+i)*fff < eps)
            {
                *(s3+i+*r3) = 0.0;
            }
            else
            {
                *(s3+i+*r3) = *(s2+i)*fff;
                fff = 1.0/fff;
                dscal_(&m, &fff, u3+(i+*r3)*m, &one);
            }
        }

        /*
         * Adjust rank
         */
        *r3 += r2;

        /*
         * Free memory
         */
        free(temp);
    }   
}

/*
 * Perform the following operations:
 * C := alpha * op(A) * op(B) + beta * C
 * where A is a m-by-m PLR matrix and B is a m-by-m low rank matrix
 * C is a m-by-m low rank matrix.
 */
static void dplrplmm(char transa, char transb, int m, double *alpha, 
    DPlrMtx *A, DPlrMtx *B, double *beta, DPlrMtx *C, double eps)
{
    int one = 1;
    double dnone = -1.0, dzero = 0.0;
    int j, len;
    double fff;

    /*
     * Test inputs
     */
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
    else if (B->type != PLR_LRMTX || C->type != PLR_LRMTX)
    {
        fprintf(stderr, "%s: invalid B or C type\n", __FUNCTION__);
        return;
    }
    else if (A->size != m || B->size != m || C->size != m)
    {
        fprintf(stderr, "%s: invalid A, B or C size\n", __FUNCTION__);
        return;   
    }

    /*
     * Quick return if possible
     */
    if (m == 0)
        return;


    /*
     * Scale C by beta
     */
    if (*beta == 0.0 || C->dlr.rank == 0)
    {
        /*
         * Set rank of C to 0
         */
        C->dlr.rank = 0;
    }
    else
    {
        if (*beta > 0.0)
        {
            dscal_(&C->dlr.rank, beta, C->dlr.sig, &one);
        }
        else
        {
            fff = -(*beta);
            dscal_(&C->dlr.rank, &fff, C->dlr.sig, &one);

            /*
             * Negative u vectors since we want to keep
             * s-values positive
             */
            len = m*C->dlr.rank;
            dscal_(&len, &dnone, C->dlr.u, &one);
        }
    }

    /*
     * If alpha = 0 or B->dlr.rank = 0
     */
    if (*alpha == 0.0 || B->dlr.rank == 0)
        return;


    /*
     * Increase the capacity of C, if necessary
     */
    if (C->dlr.capacity < C->dlr.rank + B->dlr.rank)
        dplrresizelr(C, C->dlr.rank + B->dlr.rank);

    /*
     * Start operations
     */
    if (transb == 'N' || transb == 'n')
    {
        /*
         * Form op(A) * B->dlr.u, concatenate to C->dlr.u
         */
        dplrgepdmm(transa, 'N', m, B->dlr.rank, alpha, A, B->dlr.u, m,
         &dzero, C->dlr.u+m*C->dlr.rank, m);

        /*
         * Concatenate B->dlr.v to C->dlr.v
         */
        len = m*B->dlr.rank;
        dcopy_(&len, B->dlr.v, &one, C->dlr.v+m*C->dlr.rank, &one);
    }
    else
    {
        /*
         * Form op(A) * B->dlr.v, concatenate to C->dlr.u
         */
        dplrgepdmm(transa, 'N', m, B->dlr.rank, alpha, A, B->dlr.v, m,
         &dzero, C->dlr.u+m*C->dlr.rank, m);

        /*
         * Concatenate B->dlr.u to C->dlr.v
         */
        len = m*B->dlr.rank;
        dcopy_(&len, B->dlr.u, &one, C->dlr.v+m*C->dlr.rank, &one); 
    }

    /*
     * Normalize newly concatenated u vectors in C and adjust s-values
     * accordingly. We do not normalize v vecotors since they are directly
     * copied from B, we assume they are normalized already.
     */
    for (j = 0; j < B->dlr.rank; ++j)
    {
        fff = dnrm2_(&m, C->dlr.u+m*(C->dlr.rank+j), &one);

        if (fff*B->dlr.sig[j] < eps)
        {
            C->dlr.sig[C->dlr.rank+j] = 0.0;
        }
        else
        {   
            C->dlr.sig[C->dlr.rank+j] = fff*B->dlr.sig[j];
            fff = 1.0/fff;
            dscal_(&m, &fff, C->dlr.u+m*(C->dlr.rank+j), &one);
        }
    }

    /*
     * Update rank of C
     */
    C->dlr.rank += B->dlr.rank;
}



/*
 * Perform the following operations:
 * C := alpha * op(A) * op(B) + beta * C
 * where A is a m-by-m low rank matrix and B is a m-by-m PLR matrix
 * C is a m-by-m low rank matrix.
 */
static void dplrlpmm(char transa, char transb, int m, double *alpha, 
    DPlrMtx *A, DPlrMtx *B, double *beta, DPlrMtx *C, double eps)
{
    int one = 1;
    double dnone = -1.0, dzero = 0.0;
    char trans;
    int j, len;
    double fff;

    /*
     * Test inputs
     */
    if (m < 0)
    {
        fprintf(stderr, "%s: invalid matrix size %d\n", __FUNCTION__, m);
        return;
    }
    else if (B->type != PLR_DMTX && B->type != PLR_HMTX)
    {
        fprintf(stderr, "%s: invalid B type\n", __FUNCTION__);
        return;
    }
    else if (A->type != PLR_LRMTX || C->type != PLR_LRMTX)
    {
        fprintf(stderr, "%s: invalid A or C type\n", __FUNCTION__);
        return;
    }
    else if (A->size != m || B->size != m || C->size != m)
    {
        fprintf(stderr, "%s: invalid A, B or C size\n", __FUNCTION__);
        return;   
    }

    /*
     * Quick return if possible
     */
    if (m == 0)
        return;
    

    /*
     * Scale C by beta
     */
    if (*beta == 0.0 || C->dlr.rank == 0)
    {
        /*
         * Set rank of C to 0
         */
        C->dlr.rank = 0;
    }
    else
    {
        if (*beta > 0.0)
        {
            dscal_(&C->dlr.rank, beta, C->dlr.sig, &one);
        }
        else
        {
            fff = -(*beta);
            dscal_(&C->dlr.rank, &fff, C->dlr.sig, &one);

            /*
             * Negative u vectors since we want to keep
             * s-values positive
             */
            len = m*C->dlr.rank;
            dscal_(&len, &dnone, C->dlr.u, &one);
        }
    }

    /*
     * If alpha = 0 or A->dlr.rank = 0
     */
    if (*alpha == 0.0 || A->dlr.rank == 0)
        return;


    /*
     * Increase the capacity of C, if necessary
     */
    if (C->dlr.capacity < C->dlr.rank + A->dlr.rank)
        dplrresizelr(C, C->dlr.rank + A->dlr.rank);

    /*
     * Start operations
     * First get the opposite opeation of transb
     */
    if (transb == 'N' || transb == 'n')
    {
        trans = 'T';
    }
    else
    {
        trans = 'N';
    }

    if (transa == 'N' || transa == 'n')
    {
        /*
         * Form op(B) * A->dlr.v, concatenate to C->dlr.v
         */
        dplrgepdmm(trans, 'N', m, A->dlr.rank, alpha, B, A->dlr.v, m,
         &dzero, C->dlr.v+m*C->dlr.rank, m);

        /*
         * Concatenate A->dlr.u to C->dlr.u
         */
        len = m*A->dlr.rank;
        dcopy_(&len, A->dlr.u, &one, C->dlr.u+m*C->dlr.rank, &one);
    }
    else
    {
        /*
         * Form op(B) * B->dlr.u, concatenate to C->dlr.v
         */
        dplrgepdmm(trans, 'N', m, A->dlr.rank, alpha, B, A->dlr.u, m,
         &dzero, C->dlr.v+m*C->dlr.rank, m);

        /*
         * Concatenate A->dlr.v to C->dlr.u
         */
        len = m*A->dlr.rank;
        dcopy_(&len, A->dlr.v, &one, C->dlr.u+m*C->dlr.rank, &one); 
    }

    /*
     * Normalize newly concatenated v vectors in C and adjust s-values
     * accordingly. We do not normalize u vecotors since they are directly
     * copied from B, we assume they are normalized already.
     */
    for (j = 0; j < A->dlr.rank; ++j)
    {
        fff = dnrm2_(&m, C->dlr.v+m*(C->dlr.rank+j), &one);

        if (fff*A->dlr.sig[j] < eps)
        {
            C->dlr.sig[C->dlr.rank+j] = 0.0;
        }
        else
        {   
            C->dlr.sig[C->dlr.rank+j] = fff*A->dlr.sig[j];
            fff = 1.0/fff;
            dscal_(&m, &fff, C->dlr.v+m*(C->dlr.rank+j), &one);
        }
    }

    /*
     * Update rank of C
     */
    C->dlr.rank += A->dlr.rank;
}

/*
 * Perform the matrix-matrix operations:
 * C := alpha*A*B + beta*C
 * where A, B and C are both m-by-m PLR matrices, alpha and beta are scalars.
 * Note: A, B and C must have the same PLR structure(size, level, etc).
 */
static void dplrppmm(int m, double *alpha, DPlrMtx *A, DPlrMtx *B, 
                    double *beta, DPlrMtx *C, double eps)
{
    
    char NN = 'N';
    double dzero = 0.0, done = 1.0;
    double eps2, *u, *s, *v;  
    int r, rank;


    if (m < 0)
    {
        fprintf(stderr, "%s: invalid matrix size %d\n", __FUNCTION__, m);
        return;
    }
    else if (A->type != B->type || B->type != C->type)
    {
        fprintf(stderr, "%s: A, B and C type do not match\n", __FUNCTION__);
        return;
    }
    if (A->type != PLR_DMTX && A->type != PLR_HMTX)
    {
        fprintf(stderr, "%s: invalid A, B or C type\n", __FUNCTION__);
        return;
    }
    if (A->size != m || B->size != m || C->size != m)
    {
        fprintf(stderr, "%s: invalid A, B or C size\n", __FUNCTION__);
        return;
    }

    /*
     * Quick return if possible
     */
    if (m == 0)
        return;

    if (A->type == PLR_DMTX)
    {   
        /*
         * A, B and C are dense matrices, multiply directly
         */
        dgemm_(&NN, &NN, &m, &m, &m, alpha, A->dd.d, &m, B->dd.d, &m,
                beta, C->dd.d, &m);
    }
    else
    {   
        eps2 = eps/4.0;

        /*
         * Initialize buffers to store a low rank corrections
         */
        r = MAX(MIN(A->dh.ne->dlr.rank, B->dh.sw->dlr.rank),
         MIN(A->dh.sw->dlr.rank, B->dh.ne->dlr.rank));  
        u = malloc(m/2*r*sizeof(double));
        s = malloc(r*sizeof(double));
        v = malloc(m/2*r*sizeof(double));

        /*
         * Do the operations on ne block, i.e.
         * C->dh.ne := alpha * (A->dh.nw * B->dh.ne + A->dh.ne * B->dh.se) + beta * C->dh.ne
         */
        dplrplmm('N', 'N', m/2, alpha, A->dh.nw, B->dh.ne, beta, C->dh.ne, eps2);
        dplrlpmm('N', 'N', m/2, alpha, A->dh.ne, B->dh.se, &done, C->dh.ne, eps2);


        /*   
         * Do the operations on sw block, i.e.     
         * C->dh.sw := alpha * (A->dh.sw * B->dh.nw + A->dh.se * B->dh.sw) + beta * C->dh.sw
         */ 
        dplrlpmm('N', 'N', m/2, alpha, A->dh.sw, B->dh.nw, beta, C->dh.sw, eps2);
        dplrplmm('N', 'N', m/2, alpha, A->dh.se, B->dh.sw, &done, C->dh.sw, eps2);


        /*
         * Do the operations on nw block recursively, i.e.
         * C->dh.nw := alpha * A->dh.nw * B->dh.nw + beta * C->dh.nw
         */
        dplrppmm(m/2, alpha, A->dh.nw, B->dh.nw, beta, C->dh.nw, eps2);

        /*
         * Add a low rank update, i.e.
         * C->dh.nw := alpha * A->dh.ne * B->dh.sw
         */
        rank = 0;
        dplrgellmm(m/2, A->dh.ne->dlr.rank, B->dh.sw->dlr.rank, &rank, alpha, A->dh.ne->dlr.u,  
            A->dh.ne->dlr.sig, A->dh.ne->dlr.v, B->dh.sw->dlr.u, B->dh.sw->dlr.sig,  
            B->dh.sw->dlr.v, &dzero, u, s, v, eps2);

        dplrgeprk(m/2, rank, &done, u, m/2, v, m/2, s, 1, C->dh.nw, eps2);


        /*
         * Do the operations on se block recursively, i.e.
         * C->dh.se := alpha * A->dh.se * B->dh.se + beta * C->dh.se
         */
        dplrppmm(m/2, alpha, A->dh.se, B->dh.se, beta, C->dh.se, eps2);

        /*
         * Add a low rank update, i.e.
         * C->dh.se := alpha * A->dh.sw * B->dh.ne
         */
        rank = 0;
        dplrgellmm(m/2, A->dh.sw->dlr.rank, B->dh.ne->dlr.rank, &rank, alpha, A->dh.sw->dlr.u,  
            A->dh.sw->dlr.sig, A->dh.sw->dlr.v, B->dh.ne->dlr.u, B->dh.ne->dlr.sig, 
            B->dh.ne->dlr.v, &dzero, u, s, v, eps2);
        dplrgeprk(m/2, rank, &done, u, m/2, v, m/2, s, 1, C->dh.se, eps2);

        /*
         * Free memory
         */
        free(u);
        free(v);
        free(s);
    }
}



/*
 * Perform the matrix-matrix operations:
 * C := alpha * A' * B + beta * C
 * where A, B and C are m-by-m PLR matrices, alpha and beta are scalars.
 * Note: A, B and C must have the same PLR structure(size, level, etc).
 */
static void dplrptpmm(int m, double *alpha, DPlrMtx *A, DPlrMtx *B, 
                    double *beta, DPlrMtx *C, double eps)
{
    char NN = 'N', TT = 'T';
    double dzero = 0.0, done = 1.0;
    double eps2, *u, *s, *v;  
    int r, rank;


    if (m < 0)
    {
        fprintf(stderr, "%s: invalid matrix size %d\n", __FUNCTION__, m);
        return;
    }
    else if (A->type != B->type || B->type != C->type)
    {
        fprintf(stderr, "%s: A, B and C type do not match\n", __FUNCTION__);
        return;
    }
    if (A->type != PLR_DMTX && A->type != PLR_HMTX)
    {
        fprintf(stderr, "%s: invalid A, B, or C type\n", __FUNCTION__);
        return;
    }
    if (A->size != m || B->size != m || C->size != m)
    {
        fprintf(stderr, "%s: invalid A, B or C size\n", __FUNCTION__);
        return;
    }

    /*
     * Quick return if possible
     */
    if (m == 0)
        return;

    /*
     * Start operations
     */
    if (A->type == PLR_DMTX)
    {   
        /*
         * A, B and C are dense, multiply directly
         */
        dgemm_(&TT, &NN, &m, &m, &m, alpha, A->dd.d, &m, B->dd.d, &m,
                beta, C->dd.d, &m);
    }
    else
    {   
        eps2 = eps/4.0;  

        /*
         * Initialize buffers to store a low rank update
         */
        r = MAX(MIN(A->dh.ne->dlr.rank, B->dh.ne->dlr.rank), 
            MIN(A->dh.sw->dlr.rank, B->dh.sw->dlr.rank));  
        u = malloc(m/2*r*sizeof(double));
        s = malloc(r*sizeof(double));
        v = malloc(m/2*r*sizeof(double));


        /*
         * Do the operations on ne block, i.e.
         * C->dh.ne := alpha * (A->dh.nw' * B->dh.ne + A->dh.sw' * B->dh.se) + beta * C->dh.ne
         */
        dplrplmm('T', 'N', m/2, alpha, A->dh.nw, B->dh.ne, beta, C->dh.ne, eps2);
        dplrlpmm('T', 'N', m/2, alpha, A->dh.sw, B->dh.se, &done, C->dh.ne, eps2);

        /*        
         * C->dh.sw := alpha * (A->dh.ne' * B->dh.nw + A->dh.se' * B->dh.sw) + beta * C->dh.sw
         */ 
        dplrlpmm('T', 'N', m/2, alpha, A->dh.ne, B->dh.nw, beta, C->dh.sw, eps2);
        dplrplmm('T', 'N', m/2, alpha, A->dh.se, B->dh.sw, &done, C->dh.sw, eps2);


        /*
         * Do the operations recursively on nw block, i.e.
         * C->dh.nw := alpha * A->dh.nw' * B->dh.nw + beta * C->dh.nw
         */
        dplrptpmm(m/2, alpha, A->dh.nw, B->dh.nw, beta, C->dh.nw, eps2);

        /*
         * Add a low rank update to nw block, i.e.
         * C->dh.nw := C->dh.nw + alpha * A->dh.sw' * B->dh.sw
         */
        rank = 0;
        dplrgellmm(m/2, A->dh.sw->dlr.rank, B->dh.sw->dlr.rank, &rank, alpha, A->dh.sw->dlr.v,  
            A->dh.sw->dlr.sig, A->dh.sw->dlr.u, B->dh.sw->dlr.u, B->dh.sw->dlr.sig,  
            B->dh.sw->dlr.v, &dzero, u, s, v, eps2);

        dplrgeprk(m/2, rank, &done, u, m/2, v, m/2, s, 1, C->dh.nw, eps2);


        /*
         * Do the operations recursively on se block, i.e.
         * C->dh.se := alpha * A->dh.se' * B->dh.se + beta * C->dh.se
         */
        dplrptpmm(m/2, alpha, A->dh.se, B->dh.se, beta, C->dh.se, eps2);

        /*
         * Add a low rank update to se block, i.e.
         * C->dh.se := C->dh.se + alpha * A->dh.ne' * B->dh.ne
         */
        rank = 0;
        dplrgellmm(m/2, A->dh.ne->dlr.rank, B->dh.ne->dlr.rank, &rank, alpha, A->dh.ne->dlr.v,  
            A->dh.ne->dlr.sig, A->dh.ne->dlr.u, B->dh.ne->dlr.u, B->dh.ne->dlr.sig, 
            B->dh.ne->dlr.v, &dzero, u, s, v, eps2);
        dplrgeprk(m/2, rank, &done, u, m/2, v, m/2, s, 1, C->dh.se, eps2);

        /*
         * Free memory
         */
        free(u);
        free(v);
        free(s);
    }
}


/*
 * Perform the matrix-matrix operations:
 * C := alpha * A * B' + beta * C
 * where A, B and C are m-by-m PLR matrices, alpha and beta are scalars.
 * Note: A, B and C must have the same PLR structure(size, level, etc).
 */
static void dplrpptmm(int m, double *alpha, DPlrMtx *A, DPlrMtx *B, 
                    double *beta, DPlrMtx *C, double eps)
{

    char NN = 'N', TT = 'T';
    double dzero = 0.0, done = 1.0;
    double eps2, *u, *s, *v;  
    int r, rank;


    if (m < 0)
    {
        fprintf(stderr, "%s: invalid matrix size %d\n", __FUNCTION__, m);
        return;
    }
    else if (A->type != B->type || B->type != C->type)
    {
        fprintf(stderr, "%s: A, B and C type do not match\n", __FUNCTION__);
        return;
    }
    if (A->type != PLR_DMTX && A->type != PLR_HMTX)
    {
        fprintf(stderr, "%s: invalid A, B, or C type\n", __FUNCTION__);
        return;
    }
    if (A->size != m || B->size != m || C->size != m)
    {
        fprintf(stderr, "%s: invalid A, B or C size\n", __FUNCTION__);
        return;
    }

    /*
     * Quick return if possible
     */
    if (m == 0)
        return;

    /*
     * Start operations
     */
    if (A->type == PLR_DMTX)
    {   
        /*
         * A, B and C are dense, multiply directly
         */
        dgemm_(&NN, &TT, &m, &m, &m, alpha, A->dd.d, &m, B->dd.d, &m,
                beta, C->dd.d, &m);
    }
    else
    {   
        eps2 = eps/4.0;  

        /*
         * Initialize buffers to store a low rank update
         */
        r = MAX(MIN(A->dh.ne->dlr.rank, B->dh.ne->dlr.rank), 
            MIN(A->dh.sw->dlr.rank, B->dh.sw->dlr.rank));  
        u = malloc(m/2*r*sizeof(double));
        s = malloc(r*sizeof(double));
        v = malloc(m/2*r*sizeof(double));


        /*
         * Do the operations on ne block, i.e.
         * C->dh.ne := alpha * (A->dh.nw * B->dh.sw' + A->dh.ne * B->dh.se') + beta * C->dh.ne
         */
        dplrplmm('N', 'T', m/2, alpha, A->dh.nw, B->dh.sw, beta, C->dh.ne, eps2);
        dplrlpmm('N', 'T', m/2, alpha, A->dh.ne, B->dh.se, &done, C->dh.ne, eps2);

        /*    
         * Do the operations on sw block, i.e.    
         * C->dh.sw := alpha * (A->dh.sw * B->dh.nw' + A->dh.se * B->dh.ne') + beta * C->dh.sw
         */ 
        dplrlpmm('N', 'T', m/2, alpha, A->dh.sw, B->dh.nw, beta, C->dh.sw, eps2);
        dplrplmm('N', 'T', m/2, alpha, A->dh.se, B->dh.ne, &done, C->dh.sw, eps2);

        /*
         * Do the operations recursively on nw block, i.e.
         * C->dh.nw := alpha * A->dh.nw * B->dh.nw + beta * C->dh.nw
         */
        dplrpptmm(m/2, alpha, A->dh.nw, B->dh.nw, beta, C->dh.nw, eps2);

        /*
         * Add a low rank update to nw block, i.e.
         * C->dh.nw := alpha * A->dh.ne * B->dh.ne'
         */
        rank = 0;
        dplrgellmm(m/2, A->dh.ne->dlr.rank, B->dh.ne->dlr.rank, &rank, alpha, A->dh.ne->dlr.u,  
            A->dh.ne->dlr.sig, A->dh.ne->dlr.v, B->dh.ne->dlr.v, B->dh.ne->dlr.sig,  
            B->dh.ne->dlr.u, &dzero, u, s, v, eps2);

        dplrgeprk(m/2, rank, &done, u, m/2, v, m/2, s, 1, C->dh.nw, eps2);


        /*
         * Do the operations recursively on se block, i.e.
         * C->dh.se := alpha * A->dh.se * B->dh.se' + beta * C->dh.se
         */
        dplrpptmm(m/2, alpha, A->dh.se, B->dh.se, beta, C->dh.se, eps2);

        /*
         * Add a low rank update to se block, i.e.
         * C->dh.se := alpha * A->dh.sw * B->dh.sw'
         */
        rank = 0;
        dplrgellmm(m/2, A->dh.sw->dlr.rank, B->dh.sw->dlr.rank, &rank, alpha, A->dh.sw->dlr.u,  
            A->dh.sw->dlr.sig, A->dh.sw->dlr.v, B->dh.sw->dlr.v, B->dh.sw->dlr.sig, 
            B->dh.sw->dlr.u, &dzero, u, s, v, eps2);
        dplrgeprk(m/2, rank, &done, u, m/2, v, m/2, s, 1, C->dh.se, eps2);

        /*
         * Free memory
         */
        free(u);
        free(v);
        free(s);
    }
}


/*
 * Perform the matrix-matrix operations:
 * C := alpha * A' * B' + beta * C
 * where A, B and C are m-by-m PLR matrices, alpha and beta are scalars.
 * Note: A, B and C must have the same PLR structure(size, level, etc).
 */
static void dplrptptmm(int m, double *alpha, DPlrMtx *A, DPlrMtx *B, 
                    double *beta, DPlrMtx *C, double eps)
{

    char TT = 'T';
    double dzero = 0.0, done = 1.0;
    double eps2, *u, *s, *v;  
    int r, rank;


    if (m < 0)
    {
        fprintf(stderr, "%s: invalid matrix size %d\n", __FUNCTION__, m);
        return;
    }
    else if (A->type != B->type || B->type != C->type)
    {
        fprintf(stderr, "%s: A, B and C type do not match\n", __FUNCTION__);
        return;
    }
    if (A->type != PLR_DMTX && A->type != PLR_HMTX)
    {
        fprintf(stderr, "%s: invalid A, B, or C type\n", __FUNCTION__);
        return;
    }
    if (A->size != m || B->size != m || C->size != m)
    {
        fprintf(stderr, "%s: invalid A, B or C size\n", __FUNCTION__);
        return;
    }

    /*
     * Quick return if possible
     */
    if (m == 0)
        return;

    /*
     * Start operations
     */
    if (A->type == PLR_DMTX)
    {   
        /*
         * A, B and C are dense, multiply directly
         */
        dgemm_(&TT, &TT, &m, &m, &m, alpha, A->dd.d, &m, B->dd.d, &m,
                beta, C->dd.d, &m);
    }
    else
    {   
        eps2 = eps/4.0; 

        /*
         * Initialize buffers to store a low rank update
         */ 
        r = MAX(MIN(A->dh.ne->dlr.rank, B->dh.sw->dlr.rank),
         MIN(A->dh.sw->dlr.rank, B->dh.ne->dlr.rank));  
        u = malloc(m/2*r*sizeof(double));
        s = malloc(r*sizeof(double));
        v = malloc(m/2*r*sizeof(double));


        /*
         * Do the operations on nw block, i.e.
         * C->dh.ne := alpha * (A->dh.nw' * B->dh.sw' + A->dh.sw' * B->dh.se') + beta * C->dh.ne
         */
        dplrplmm('T', 'T', m/2, alpha, A->dh.nw, B->dh.sw, beta, C->dh.ne, eps2);
        dplrlpmm('T', 'T', m/2, alpha, A->dh.sw, B->dh.se, &done, C->dh.ne, eps2);

        /*    
         * Do the operations on se block, i.e.    
         * C->dh.sw := alpha * (A->dh.ne' * B->dh.nw' + A->dh.se' * B->dh.ne') + beta * C->dh.sw
         */ 
        dplrlpmm('T', 'T', m/2, alpha, A->dh.ne, B->dh.nw, beta, C->dh.sw, eps2);
        dplrplmm('T', 'T', m/2, alpha, A->dh.se, B->dh.ne, &done, C->dh.sw, eps2);


        /*
         * Do the operations recursively on nw block, i.e.
         * C->dh.nw := alpha * A->dh.nw' * B->dh.nw' + beta * C->dh.nw
         */
        dplrptptmm(m/2, alpha, A->dh.nw, B->dh.nw, beta, C->dh.nw, eps2);

        /*
         * Add a low rank update to nw block, i.e.
         * C->dh.nw := alpha * A->dh.sw' * B->dh.ne'
         */
        rank = 0;
        dplrgellmm(m/2, A->dh.sw->dlr.rank, B->dh.ne->dlr.rank, &rank, alpha, A->dh.sw->dlr.v,  
            A->dh.sw->dlr.sig, A->dh.sw->dlr.u, B->dh.ne->dlr.v, B->dh.ne->dlr.sig,  
            B->dh.ne->dlr.u, &dzero, u, s, v, eps2);

        dplrgeprk(m/2, rank, &done, u, m/2, v, m/2, s, 1, C->dh.nw, eps2);


        /*
         * Do the operations recursively on se block, i.e.
         * C->dh.se := alpha * A->dh.se' * B->dh.se' + beta * C->dh.se
         */
        dplrptptmm(m/2, alpha, A->dh.se, B->dh.se, beta, C->dh.se, eps2);

        /*
         * Add a low rank update to se block, i.e.
         * C->dh.se := alpha * A->dh.ne' * B->dh.sw'
         */
        rank = 0;
        dplrgellmm(m/2, A->dh.ne->dlr.rank, B->dh.sw->dlr.rank, &rank, alpha, A->dh.ne->dlr.v,  
            A->dh.ne->dlr.sig, A->dh.ne->dlr.u, B->dh.sw->dlr.v, B->dh.sw->dlr.sig, 
            B->dh.sw->dlr.u, &dzero, u, s, v, eps2);
        dplrgeprk(m/2, rank, &done, u, m/2, v, m/2, s, 1, C->dh.se, eps2);

        /*
         * Free memory
         */
        free(u);
        free(v);
        free(s);
    }
}

/*
 * Perform one of the matrix-matrix operations:
 * C := alpha*op(A)*op(B) + beta*C
 * where B and C are both m-by-m PLR matrices, alpha and beta are scalars, A is a PLR matrix.
 * TRANSA = 'N' or 'n'	op(A) := A
 * TRANSA = 'T' or 't'  op(A) := A**T
 * TRANSB = 'N' or 'n'	op(B) := B
 * TRANSB = 'T' or 't'  op(B) := B**T
 * Note: A, B and C must have the same PLR structure(size, level, etc).
 */
void dplrgeppmm(char transa, char transb, int m, double *alpha, DPlrMtx *A, 
	DPlrMtx *B, double *beta, DPlrMtx *C, double eps)
{
    
    /*
     * Start operations
     * We do not check inputs since subroutines 
     * will check them
     */
    if (transa == 'N' || transa == 'n')
    {
        if (transb == 'N' || transb == 'n')
        {
            dplrppmm(m, alpha, A, B, beta, C, eps);
        }
        else
        {
            dplrpptmm(m, alpha, A, B, beta, C, eps);
        }
    }
    else
    {
        if (transb == 'N' || transb == 'n')
        {
            dplrptpmm(m, alpha, A, B, beta, C, eps);
        }
        else
        {
            dplrptptmm(m, alpha, A, B, beta, C, eps);
        }
    }
}

 /***********************************************************************
 *
 * END: dplrppmm.c
 *
 ***********************************************************************/

