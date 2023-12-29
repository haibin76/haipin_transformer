#include <stdlib.h>
#include <math.h>
#if !_WIN32
#include <time.h>
#endif
#include "kernal.h"

float* matrix_create(int batch_num, int height, int width)
{
    float* matrix = new float[batch_num * height * width];
    if (!matrix)
        return NULL;

#if !_WIN32
    srand(time(NULL));  // 设置随机数种子
#endif

    for (int b = 0; b < batch_num; b++)
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++) {
                matrix[(b * height + y) * width + x] = (float)(((rand() % 0xFF) - 128) * 1.0 / 256);
            }

    return matrix;
}

void matrix_delete(void* matrix)
{
    delete[] matrix;
    return;
}

void matrix_add(int batch_num, float* matrix_a, float* matrix_b, float* matrix_c, char* mask, int height, int width)
{
    for(int b = 0; b < batch_num; b++)
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++) {
                int address = (b * height + y) * width + x;
                if (mask && mask[address] == 0)
                    matrix_c[address] = 0;
                else
                    matrix_c[address] = matrix_a[address] + matrix_b[address];
            }

    return;
}

void matrix_multi(int batch_num, float* matrix_a, int a_height, int a_width, float* matrix_b, int b_width, float* matrix_c)
{
    for (int b = 0; b < batch_num; b++) {
        int batch_offset_a = b * a_height * a_width;
        int batch_offset_b = b * a_width * b_width;

        for (int y = 0; y < a_height; y++)
            for (int x = 0; x < b_width; x++) {
                double sum = 0.0;
                for (int k = 0; k < a_width; k++)
                    sum += matrix_a[y * a_width + k] * matrix_b[k * b_width + x];

                matrix_c[y * b_width + x] = (float)sum;
            }
    }

    return;
}

void matrix_multi_without_transpose(int batch_num, float* matrix_a, int a_height, int a_num, int a_width,
                                                       float* matrix_b, int b_height, int b_width,
                                                       char* mask, double scale, float* matrix_c)
{
    for (int b = 0; b < batch_num; b++) {
        int batch_offset_a = b * a_height * a_width;
        int batch_offset_b = b * a_width  * b_height;
        int batch_offset_c = b * a_height * b_height;

        for (int y = 0; y < a_height; y++)
            for (int x = 0; x < b_height; x++) {
                if (mask && (!mask[batch_offset_c + y * b_height + x]))
                    matrix_c[batch_offset_c + y * b_height + x] = (float)(1/10000.0);
                else {
                    double sum = 0.0;
                    for (int k = 0; k < a_num; k++)
                        sum += matrix_a[batch_offset_a + y * a_width + k] * matrix_b[batch_offset_b + x * b_width + k];

                    matrix_c[batch_offset_c + y * b_height + x] = (float)(sum/ scale);
                }
            }
    }

    return;

}

void add_layer_norm(int batch_num, float* matrix_a, int height, int width, float* matrix_c, char* gamme, char* beta)
{
    for (int b = 0; b < batch_num; b++) {
        int batch_offse = height * width;
        for (int y = 0; y < height; y++) {
            //this meathod, calculate every row
            double mean = 0.0, var = 0.0, tmp;

            //mean
            for (int x = 0; x < width; x++) {
                mean += matrix_a[batch_offse + y * width + x];
            }
            mean = mean / width;

            //var
            for (int x = 0; x < width; x++) {
                tmp = matrix_a[batch_offse + y * width + x] - mean;
                tmp = tmp * tmp;
                var += tmp;
            }
            var = var / width;
            var += 1e-3;//avoid to too small
            var = sqrt(var);

            for (int x = 0; x < width; x++) {
                tmp = (matrix_a[batch_offse + y * width + x] - mean) / var;
                matrix_c[batch_offse + y * width + x] = (float)(gamme[x] * tmp + beta[x]);
            }
        }
    }

    return;
}

void softmax(int batch_num, float* matrix_a, int height, int width, float* matrix_c)
{
    //if the vector [a, b, c], if we calculate softmax with normal meathod, the result perhaps overflow
    //we use the math as flow:softmax(a) = exp(a/k) / (exp(a/k) + exp(b-k) + exp(c-k) ...)

    for (int b = 0; b < batch_num; b++) {
        for (int y = 0; y < height; y++) {
            //find the max num int the vector
            double max_value = 0.0, sum = 0.0;

            for (int x = 0; x < width; x++) {
                if (max_value < matrix_a[(b * height + y) * width + x])
                    max_value = matrix_a[(b * height + y) * width + x];
            }
            max_value -= 1.0;

            for (int x = 0; x < width; x++) {
                sum += exp(matrix_a[(b * height + y) * width + x] - max_value);
            }

            for (int x = 0; x < width; x++) {
                matrix_c[(b * height + y) * width + x] = (float)(exp(matrix_a[(b * height + y) * width + x] - max_value) / sum);
            }

        }
    }

    return;
}