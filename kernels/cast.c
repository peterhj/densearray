#include <xmmintrin.h>
#include <smmintrin.h>

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

void densearray_kernel_cast_1d_u8_to_f32(
    size_t dim,
    const uint8_t *src,
    float *dst)
{
  for (size_t idx = 0; idx < dim; idx++) {
    dst[idx] = (float)(src[idx]);
  }
}

void densearray_kernel_round_clamp_1d_f32_to_u8_sse2(
    size_t dim,
    const float *src,
    uint8_t *dst)
{
  __m128 lower = _mm_set_ss(0.0f);
  __m128 upper = _mm_set_ss(255.0f);
  __m128 one_half = _mm_set_ss(0.5f);
  __m128 one = _mm_set_ss(1.0f);
  for (size_t idx = 0; idx < dim; idx++) {
    __m128 x = _mm_load_ss(&src[idx]);
    __m128 y = _mm_min_ss(_mm_max_ss(lower, x), upper);
    __m128 u = _mm_add_ps(y, one_half);
    __m128 v = _mm_cvtepi32_ps(_mm_cvttps_epi32(u));
    // XXX: Can ignore round-to-zero issues since `v` is always in the range
    // [0.0f, 255.0f].
    /*__m128 w = _mm_sub_ps(v, _mm_and_ps(_mm_cmplt_ps(u, v), one));*/
    dst[idx] = (uint8_t)(_mm_cvtss_f32(v));
  }
}

void densearray_kernel_round_clamp_1d_f32_to_u8_sse4(
    size_t dim,
    const float *src,
    uint8_t *dst)
{
  __m128 lower = _mm_set_ss(0.0f);
  __m128 upper = _mm_set_ss(255.0f);
  for (size_t idx = 0; idx < dim; idx++) {
    __m128 x = _mm_load_ss(&src[idx]);
    __m128 y = _mm_min_ss(_mm_max_ss(lower, x), upper);
    __m128 u = _mm_round_ps(y, _MM_FROUND_NINT);
    dst[idx] = (uint8_t)(_mm_cvtss_f32(u));
  }
}

void densearray_kernel_clamp_1d_f32_sse2(
    size_t dim,
    const float *src,
    float *dst,
    float lowerf,
    float upperf)
{
  __m128 lower = _mm_set_ss(lowerf);
  __m128 upper = _mm_set_ss(upperf);
  for (size_t idx = 0; idx < dim; idx++) {
    __m128 x = _mm_load_ss(&src[idx]);
    __m128 y = _mm_min_ss(_mm_max_ss(lower, x), upper);
    _mm_store_ss(&dst[idx], y);
  }
}
