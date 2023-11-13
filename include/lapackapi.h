/***********************************************************************
 *
 * NAME: lapackapi.h
 *
 * DESC: Prototypes for BLAS and LAPACK routines, definitions for
 *   	 real numbers, etc. This is used to easily switch different
 * 		 LAPACK/BLAS implementations.
 *
 * AUTH: Yang Xinshuo
 *
 * DATE: 20140624
 *
 **********************************************************************/

#ifndef _LAPACKAPI_H_
#define _LAPACKAPI_H_

/*
 * Either USE_ACML or USE_MKL must be defined
 */

#if defined(USE_MKL)
#include "lapackapi_mkl.h"

#elif defined(USE_NETLIB)
#include "lapackapi_netlib.h"

#else
#error "Neither USE_MKL nor USE_NTELIB are #defined"

#endif

#endif

/***********************************************************************
 *
 * END: lapackapi.h
 *
 **********************************************************************/
