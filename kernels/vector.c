#include "lib.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

void DENSEARRAY_SYMBOL(set_scalar_f32)(
    float *dst,
    size_t dim,
    float c)
{
  #pragma omp parallel for
  for (size_t idx = 0; idx < dim; idx++) {
    dst[idx] = c;
  }
}

void DENSEARRAY_SYMBOL(copy_f32)(
    float *dst,
    size_t dim,
    const float *src)
{
  #pragma omp parallel for
  for (size_t idx = 0; idx < dim; idx++) {
    dst[idx] = src[idx];
  }
}

void DENSEARRAY_SYMBOL(square_f32)(
    float *dst,
    size_t dim)
{
  #pragma omp parallel for
  for (size_t idx = 0; idx < dim; idx++) {
    float y = dst[idx];
    dst[idx] = y * y;
  }
}

void DENSEARRAY_SYMBOL(sqrt_f32)(
    float *dst,
    size_t dim)
{
  #pragma omp parallel for
  for (size_t idx = 0; idx < dim; idx++) {
    float y = dst[idx];
    dst[idx] = sqrtf(y);
  }
}

void DENSEARRAY_SYMBOL(reciprocal_f32)(
    float *dst,
    size_t dim)
{
  #pragma omp parallel for
  for (size_t idx = 0; idx < dim; idx++) {
    float y = dst[idx];
    dst[idx] = 1.0f / y;
  }
}

void DENSEARRAY_SYMBOL(add_scalar_f32)(
    float *dst,
    size_t dim,
    float c)
{
  #pragma omp parallel for
  for (size_t idx = 0; idx < dim; idx++) {
    dst[idx] += c;
  }
}

void DENSEARRAY_SYMBOL(scale_f32)(
    float *dst,
    size_t dim,
    float c)
{
  #pragma omp parallel for
  for (size_t idx = 0; idx < dim; idx++) {
    dst[idx] *= c;
  }
}

void DENSEARRAY_SYMBOL(div_scalar_f32)(
    float *dst,
    size_t dim,
    float c)
{
  #pragma omp parallel for
  for (size_t idx = 0; idx < dim; idx++) {
    dst[idx] /= c;
  }
}

void DENSEARRAY_SYMBOL(elem_mult_f32)(
    float *dst,
    size_t dim,
    const float *xs)
{
  #pragma omp parallel for
  for (size_t idx = 0; idx < dim; idx++) {
    dst[idx] *= xs[idx];
  }
}

void DENSEARRAY_SYMBOL(elem_div_f32)(
    float *dst,
    size_t dim,
    const float *xs)
{
  #pragma omp parallel for
  for (size_t idx = 0; idx < dim; idx++) {
    dst[idx] /= xs[idx];
  }
}

void DENSEARRAY_SYMBOL(elem_ldiv_f32)(
    float *dst,
    size_t dim,
    const float *xs)
{
  #pragma omp parallel for
  for (size_t idx = 0; idx < dim; idx++) {
    dst[idx] = xs[idx] / dst[idx];
  }
}

void DENSEARRAY_SYMBOL(vector_add_f32)(
    float *dst,
    size_t dim,
    const float *xs,
    float c)
{
  #pragma omp parallel for
  for (size_t idx = 0; idx < dim; idx++) {
    dst[idx] += c * xs[idx];
  }
}

void DENSEARRAY_SYMBOL(vector_average_f32)(
    float *dst,
    size_t dim,
    const float *xs,
    float c)
{
  #pragma omp parallel for
  for (size_t idx = 0; idx < dim; idx++) {
    dst[idx] += c * (xs[idx] - dst[idx]);
  }
}
