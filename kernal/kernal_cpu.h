#ifndef __KERNAL_CPU__
#define __KERNAL_CPU__

float* matrix_create_cpu(int batch_num, int height, int width);

void matrix_delete_cpu(void* matrix);

void matrix_add_cpu(int batch_num, float* matrix_a, float* matrix_b, float* matrix_c, char* mask, int height, int width);

//assume that a_width == b_height
void matrix_multi_cpu(int batch_num, float* matrix_a, int a_height, int a_width, float* matrix_b, int b_width, float* matrix_c);

void matrix_multi_without_transpose_cpu(int batch_num, float* matrix_a, int a_height, int a_num, int a_width,
                                                       float* matrix_b, int b_height, int b_width,
                                                       char* mask, double scale, float* matrix_c);

void add_layer_norm_cpu(int batch_num, float* matrix_a, int height, int width, float* matrix_c, char* gamme, char* beta);

void softmax_cpu(int batch_num, float* matrix_a, int height, int width, float* matrix_c);
#endif
