#ifndef __KERNAL_CPU__
#define __KERNAL_CPU__

float* matrix_create(int batch_num, int height, int width);

void matrix_delete(void* matrix);

void matrix_add(int batch_num, float* matrix_a, float* matrix_b, float* matrix_c, char* mask, int height, int width);

//assume that a_width == b_height
void matrix_multi(int batch_num, float* matrix_a, int a_height, int a_width, float* matrix_b, int b_width, float* matrix_c);

void matrix_multi_without_transpose(int batch_num, float* matrix_a, int a_height, int a_num, int a_width,
                                                       float* matrix_b, int b_height, int b_width,
                                                       char* mask, double scale, float* matrix_c);

void add_layer_norm(int batch_num, float* matrix_a, int height, int width, float* matrix_c, char* gamme, char* beta);

void softmax(int batch_num, float* matrix_a, int height, int width, float* matrix_c);

#if !__WIN32
bool init_gpu(int device_id);
#endif

#endif
