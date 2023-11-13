/***********************************************************************
 *
 * NAME: dplrempty.c
 *
 * DESC: Create a empty PLR matrix
 *
 * AUTH: Yang Xinshuo
 *
 * DATE: 20140422
 *
 **********************************************************************/
#include "../include/dplr.h"

/*
 * Create a empty matrix with the same decompostion and 
 * size of the given matrix.
 */
void dplrempty(DPlrMtx *A, DPlrMtx *B)
{
    if (A->type != PLR_DMTX && A->type != PLR_HMTX)
    {
        fprintf(stderr, "%s: A must be a dense or hierarchical matrix \n",
             __FUNCTION__);
        return;
    }

    if (A->type == PLR_DMTX) 
    {
        dplrclr(B);
        dplrinid(A->size, B);
    }
    else 
    {
        dplrclr(B);
        dplrinih(A->size, B);
        dplrinilr(A->size/2, B->dh.ne, 0);
        dplrinilr(A->size/2, B->dh.sw, 0);

        dplrempty(A->dh.nw, B->dh.nw);
        dplrempty(A->dh.se, B->dh.se);
    }

    return;
}

/***********************************************************************
 *
 * END: dplrempty.c
 *
 **********************************************************************/
