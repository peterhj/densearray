#ifndef __DENSEARRAY_KERNELS_LIB_H__
#define __DENSEARRAY_KERNELS_LIB_H__

#ifndef DENSEARRAY_OMP
#define DENSEARRAY_SYMBOL(name) densearray_ ## name
#else
#define DENSEARRAY_SYMBOL(name) densearray_omp_ ## name
#endif

#endif
