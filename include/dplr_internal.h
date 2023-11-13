/***********************************************************************
 *
 * NAME: dplr_internal.h
 *
 * DESC: Internal definitions for dplr.h. Include that file instead of
 *       this one.
 *
 * AUTH: Yang Xinshuo
 *
 * DATE: 20140826
 *
 **********************************************************************/

/************************************************************************
 *
 * PLR matrix data structures
 *
 ************************************************************************/


#ifndef _DPLR_INTERNAL_H_
#define _DPLR_INTERNAL_H_

#ifndef _DPLR_H_
# error "Never include dplr_internal.h directly; use dplr.h instead."
#endif

 
/*
 * Matrix types
 */
typedef enum
{
    PLR_UMTX=0,         /* Undefined matrix type (not yet populated) */
    PLR_HMTX,           /* A hierachical matrix */
    PLR_DMTX,           /* A dense matrix */
    PLR_LRMTX           /* A low rank matrix */
} PlrMtxType;            

/*
 * Forward declaration of generic matrix type
 */
typedef struct DPlrMtxStruct DPlrMtx;

/*
 * Hierachical matrix, i.e. 2-by-2 block matrix
 */
typedef struct
{
    DPlrMtx *nw, *ne, *se, *sw; /* Pointers to each block */
} DPlrHMtx;

/*
 * Dense matrix
 */
typedef struct
{
    double *d;           /* size-by-size dense matrix */
} DPlrDMtx;

/*
 * Low rank matrix
 */
typedef struct
{
    int rank;                   /* Rank of this matrix */
    int capacity;               /* Max rank of this matrix (size of buffers) */
    double *u, *v;         /* Data buffers */
    double *sig;
} DPlrLRMtx;

/*
 * Generic matrix
 */
struct DPlrMtxStruct
{
    PlrMtxType type;            /* What type of matrix is this? */
    int size;                   /* Size of this matrix */

    /* Only one of these records is populated, depending on the type */
    union
    {
        DPlrHMtx  dh;
        DPlrDMtx  dd;
        DPlrLRMtx dlr;
    };
};


/************************************************************************
 *
 * Private routines used internally by the PLR library
 *
 ************************************************************************/

/*
 * Initialze dense A with indicated size
 */
void dplrinid(int m, DPlrMtx *A);

/*
 * Initialze low rank A with indicated size and capacity
 */
void dplrinilr(int m, DPlrMtx *A, int capacity);

/*
 * Initialze hierarchical A with indicated size
 */
void dplrinih(int m, DPlrMtx *A);

/*
 * Increase the capacity of low rank A. capacity is a lower bound --
 * the routine may increase the capacity by more than is requested.
 */
void dplrresizelr(DPlrMtx *A, int capacity);

#endif

/***********************************************************************
 *
 * END: dplr_internal.h
 *
 **********************************************************************/
