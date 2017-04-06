#include <stdint.h>
#include <stdlib.h>

void densearray_kernel_elem_increduce_sum_f32(
    size_t dim,
    const float *x,
    float *sum)
{
  for (size_t idx = 0; idx < dim; idx++) {
    sum[idx] += x[idx];
  }
}

void densearray_kernel_elem_increduce_mean_f32(
    size_t dim,
    size_t prev_count,
    const float *x,
    float *mean)
{
  float n = (float)(prev_count + 1);
  float inv_n = 1.0f / n;
  for (size_t idx = 0; idx < dim; idx++) {
    mean[idx] += inv_n * (x[idx] - mean[idx]);
  }
}

void densearray_kernel_elem_increduce_stats2_f32(
    size_t dim,
    size_t prev_count,
    const float *x,
    float *mean,
    float *uvar)
{
  float n = (float)(prev_count + 1);
  float inv_n = 1.0f / n;
  for (size_t idx = 0; idx < dim; idx++) {
    float x_i = x[idx];
    float prev_mean_i = mean[idx];
    mean[idx] += inv_n * (x_i - prev_mean_i);
    uvar[idx] += (x_i - prev_mean_i) * (x_i - mean[idx]);
  }
}

void densearray_kernel_elem_increduce_stats4_f32(
    size_t dim,
    size_t prev_count,
    const float *x,
    float *mean,
    float *uvar,
    float *ucm3,
    float *ucm4)
{
  // See: X. Meng, "Simpler Online Updates for Arbitrary-Order Central Moments", 2015 (arXiv:1510.04923).
  float n = (float)(prev_count + 1);
  float inv_n = 1.0f / n;
  for (size_t idx = 0; idx < dim; idx++) {
    float x_i = x[idx];
    float prev_mean_i = mean[idx];
    float delta_i = x_i - prev_mean_i;
    float delta_i_over_n = inv_n * delta_i;
    mean[idx] = delta_i_over_n;
    uvar[idx] += delta_i * (x_i - mean[idx]);
    ucm3[idx] += -3.0f * delta_i_over_n * uvar[idx] + delta_i * (delta_i * delta_i - delta_i_over_n * delta_i_over_n);
    ucm4[idx] += -4.0f * delta_i_over_n * ucm3[idx] - 6.0f * delta_i_over_n * delta_i_over_n * uvar[idx] + delta_i * (delta_i * delta_i * delta_i - delta_i_over_n * delta_i_over_n * delta_i_over_n);
  }
}

void densearray_kernel_elem_postreduce_var_f32(
    size_t dim,
    size_t count,
    float *uvar)
{
  float n = (float)(count);
  float scale = 1.0 / (n - 1.0f);
  for (size_t idx = 0; idx < dim; idx++) {
    uvar[idx] *= scale;
  }
}

void densearray_kernel_elem_postreduce_cm3_f32(
    size_t dim,
    size_t count,
    float *ucm3)
{
  // See: B. Klemens, Appendix M of "Modeling with Data", 2008.
  float n = (float)(count);
  float scale = (n / (n - 1.0f)) / (n - 2.0f);
  for (size_t idx = 0; idx < dim; idx++) {
    ucm3[idx] *= scale;
  }
}

void densearray_kernel_elem_postreduce_cm4_f32(
    size_t dim,
    size_t count,
    const float *uvar,
    float *ucm4)
{
  // See: B. Klemens, Appendix M of "Modeling with Data", 2008.
  float n = (float)(count);
  float scale4 = (n * (n * (n - 1.0f) * (n - 1.0f) + (6.0f * n - 9.0f))) / ((n - 1.0f) * (n - 1.0f) * (n - 1.0f) * (n * n - 3.0f * n + 3.0f));
  float scale2 = (n * (6.0f * n - 9.0f)) / ((n - 1.0f) * (n - 1.0f) * (n - 1.0f) * (n * n - 3.0f * n + 3.0f));
  for (size_t idx = 0; idx < dim; idx++) {
    ucm4[idx] = scale4 * ucm4[idx] + scale2 * uvar[idx];
  }
}
