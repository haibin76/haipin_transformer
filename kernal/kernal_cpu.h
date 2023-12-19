#ifndef __KERNAL_CPU__
#define __KERNAL_CPU__

float* matrix_create_cpu(int height, int width);

void matrix_add_cpu(float* a, float* b, float* c, int a_height, int a_width);

void matrix_multi_cpu(float* a, float* b, float *c, int a_height, int a_width, int b_width);

void add_layer_norm_cpu(float* in, float* out, char* gamme, char* beta, int height, int width);

void softmax_cpu(float* in, float* out, double* buffer, int height, int width);

void scale_dot_product_attention(float* q, int q_height, int q_width, float* k, int k_height, int k_width,
                            float* v, int v_height, int v_width, char* mask, int dim,
                            float* out, int* out_height, int* out_width);

#endif
