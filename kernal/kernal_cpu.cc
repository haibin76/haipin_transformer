#include <stdlib.h>
#include <math.h>
#if !_WIN32
#include <time.h>
#endif
#include "kernal_cpu.h"

float* matrix_create_cpu(int height, int width)
{
    float* matrix = new float[height * width];
    if (!matrix)
        return NULL;

#if !_WIN32
    srand(time(NULL));  // 设置随机数种子
#endif

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            matrix[y * width + x] = (float)(((rand() % 0xFF) - 128) * 1.0 / 256);
        }
    }

    return matrix;
}

void matrix_add_cpu(float* a, float* b, float* c, int height, int width)
{
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            c[y * width + x] = a[y * width + x] + b[y * width + x];
        }

    return;
}

void matrix_multi_cpu(float* a, float* b, float* c, int a_height, int a_width, int b_width)
{
    for(int y = 0; y < a_height; y++)
        for (int x = 0; x < b_width; x++) {
            float sum = 0.0;
            for (int k = 0; k < a_width; k++)
                sum += a[y * a_width + k] * b[k * b_width + x];

            c[y * b_width + x] = sum;
        }

    return;
}

void add_layer_norm_cpu(float* in, float* out, char* gamme, char* beta, int height, int width)
{
    for (int y = 0; y < height; y++) {
        double mean = 0.0, var = 0.0, tmp;

        //mean
        for (int x = 0; x < width; x++) {
            mean += in[y * width + x];
        }
        mean = mean / width;

        //var
        for (int x = 0; x < width; x++) {
            tmp = in[y * width + x] - mean;
            tmp = tmp * tmp;
            var += tmp;
        }
        var = var / width;
        var += 1e-3;//avoid to too small
        var = sqrt(var);

        for (int x = 0; x < width; x++) {
            tmp = (in[y * width + x] - mean) / var;
            out[y * width + x] = (float)(gamme[x] * tmp + beta[x]);
        }

    }

    return;
}

void softmax_cpu(float* in, float* out, double* buffer, int height, int width)
{
    for (int y = 0; y < height; y++) {
        double sum = 0.0;

        for (int x = 0; x < width; x++) {
            buffer[x] = exp(in[y * width + x]);
            sum += buffer[x];
        }

        for (int x = 0; x < width; x++) {
            out[y * width + x] = (float)(buffer[x] / sum);
        }

    }

    return;
}

void scale_dot_product_attention(float* q, int q_height, int q_width, float* k, int k_height, int k_width,
                                 float* v, int v_height, int v_width, char* mask, int dim,
                                 float* out, int* out_height, int* out_width)
{
    double scale = sqrt(dim);

    //now, we think:all parameters are legal, dont need to check
    //#0 calc the Q * K^T, for performance, we dont tranpose the matrix
    for(int y = 0; y < q_height; y++)
        for (int x = 0; x < k_height; x++) {
            double sum = 0.0;
            for (int i = 0; i < q_width; i++)
                sum += (q[y * q_width + i] * k[x * k_width + i]);

            if (!mask[y * k_height + x])
                out[y * k_height + x] = -10000.0;
            else
                out[y * k_height + x] = (float)(sum/scale);
        }

    //#1 softmax(out)
    for (int y = 0; y < q_height; y++)
        softmax_cpu(out, q, (double*)k, q_height, k_height);

    //#2 MatMul(softmax(q*k^t) * v)
    matrix_multi_cpu(q, v, out, q_height, k_height, v_width);

    *out_height = q_height;
    *out_width = v_width;

    return;
}