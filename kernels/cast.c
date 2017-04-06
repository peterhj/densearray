#include <xmmintrin.h>

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
  /*__m128 lower = _mm_set1_ss(0.0f);
  __m128 upper = _mm_set1_ss(255.0f);
  size_t idx = 0;
  for ( ; idx < dim / 4 * 4; idx += 4) {
    __m128 x = _mm_loadu_ps(&src[idx]);
    __m128 y = _mm_min_ps(_mm_max_ps(lower, x), upper);
    // TODO
  }
  for ( ; idx < dim; idx++) {
  }*/
  __m128 lower = _mm_set_ss(0.0f);
  __m128 upper = _mm_set_ss(255.0f);
  for (size_t idx = 0; idx < dim; idx++) {
    __m128 x = _mm_load_ss(&src[idx]);
    __m128 y = _mm_min_ss(_mm_max_ss(lower, x), upper);
    float z = 0.0f;
    _mm_store_ss(&z, y);
    dst[idx] = (uint8_t)(floorf(z + 0.5f));
  }
}
