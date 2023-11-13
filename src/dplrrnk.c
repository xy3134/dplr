/***********************************************************************
 *
 * NAME: dplrrnk.c
 *
 * DESC: Populate the maximum ranks of a PLR matrix at each level
 *
 * AUTH: Xinshuo Yang
 *
 * DATE: 20150327
 *
 ***********************************************************************/

#include "../include/dplr.h"
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

/*
 * Populate the array ranks with the maximum rank required at each
 * level to store the m-by-m PLR matrix A. ranks should be an array of
 * length nlvl, where nlvl is the level of PLR decomposition. On
 * entry, all elements of ranks should be initialized to -1.
 */
void dplrrnk(int m, DPlrMtx *A, int *ranks)
{
	if (m < 0)
	{
		fprintf(stderr, "%s: invalid matrix size %d\n", __FUNCTION__, m);
		return;		
	}
	if (A->type != PLR_DMTX && A->type != PLR_HMTX)
	{
		fprintf(stderr, "%s: invalid A type\n", __FUNCTION__);
		return;	
	}
	else if (A->size != m)
	{
		fprintf(stderr, "%s: invalid A size\n", __FUNCTION__);
		return;
	}

    if (A->type == PLR_HMTX)
    {
        /*
         * Update rank from current level
         */
        int r = MAX(A->dh.ne->dlr.rank, A->dh.sw->dlr.rank);

        if (*ranks < r) *ranks = r;

        /*
         * Get rank from next level 
         */
        if (A->dh.nw->type != PLR_DMTX && A->dh.se->type != PLR_DMTX)
        {
            dplrrnk(m/2, A->dh.nw, ranks+1);
            dplrrnk(m/2, A->dh.se, ranks+1);
        }
    }
}

/***********************************************************************
 *
 * END: dplrrnk.c
 *
 **********************************************************************/
