#include <math.h>
#include <stdint.h>
#include <stdlib.h>

void densearray_square_f32(
    float *dst,
    size_t dim)
{
  for (size_t idx = 0; idx < dim; idx++) {
    float y = dst[idx];
    dst[idx] = y * y;
  }
}

void densearray_sqrt_f32(
    float *dst,
    size_t dim)
{
  for (size_t idx = 0; idx < dim; idx++) {
    float y = dst[idx];
    dst[idx] = sqrtf(y);
  }
}

void densearray_reciprocal_f32(
    float *dst,
    size_t dim)
{
  for (size_t idx = 0; idx < dim; idx++) {
    float y = dst[idx];
    dst[idx] = 1.0f / y;
  }
}

void densearray_add_scalar_f32(
    float *dst,
    size_t dim,
    float c)
{
  for (size_t idx = 0; idx < dim; idx++) {
    dst[idx] += c;
  }
}

void densearray_scale_f32(
    float *dst,
    size_t dim,
    float c)
{
  for (size_t idx = 0; idx < dim; idx++) {
    dst[idx] *= c;
  }
}

void densearray_div_scalar_f32(
    float *dst,
    size_t dim,
    float c)
{
  for (size_t idx = 0; idx < dim; idx++) {
    dst[idx] /= c;
  }
}

void densearray_elem_mult_f32(
    float *dst,
    size_t dim,
    const float *xs)
{
  for (size_t idx = 0; idx < dim; idx++) {
    dst[idx] *= xs[idx];
  }
}

void densearray_elem_div_f32(
    float *dst,
    size_t dim,
    const float *xs)
{
  for (size_t idx = 0; idx < dim; idx++) {
    dst[idx] /= xs[idx];
  }
}

void densearray_elem_ldiv_f32(
    float *dst,
    size_t dim,
    const float *xs)
{
  for (size_t idx = 0; idx < dim; idx++) {
    dst[idx] = xs[idx] / dst[idx];
  }
}

void densearray_vector_add_f32(
    float *dst,
    size_t dim,
    const float *xs,
    float c)
{
  for (size_t idx = 0; idx < dim; idx++) {
    dst[idx] += c * xs[idx];
  }
}

void densearray_vector_average_f32(
    float *dst,
    size_t dim,
    const float *xs,
    float c)
{
  for (size_t idx = 0; idx < dim; idx++) {
    dst[idx] += c * (xs[idx] - dst[idx]);
  }
}
