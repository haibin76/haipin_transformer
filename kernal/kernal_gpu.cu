#include <stdlib.h>
#include <math.h>
#if !_WIN32
#include <sys/time.h>
#endif

#include <cuda_runtime.h>
#include "kernal.h"

//在打开GPU的之前，需要同时初始化GPU，查看GPU的一些性能和参数配置
bool init_gpu(int device_id)
{
    //查看有几个GPU卡
    printf("%s Starting...\n", device_id);
    int device_count = 0;
    cudaError_t error_id = cudaGetDeviceCount(&device_count);
    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    if (device_count == 0) {
        printf("There are no available device(s) that support CUDA\n");
    } else {
        printf("Detected %d CUDA Capable device(s)\n", device_count);
        if (device_count <device_id) {
            printf("the input parameters device_id:%d error\n", device_id);
        }
    }

    int dev, driverVersion = 0, runtimeVersion = 0;
    dev = device_id;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Device %d: \"%s\"\n", dev, deviceProp.name);
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf(" CUDA Driver Version / Runtime Version %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
    printf(" CUDA Capability Major/Minor version number: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf(" Total amount of global memory: %.2f MBytes (%llu bytes)\n", (float)deviceProp.totalGlobalMem/(pow(1024.0,3)), (unsigned long long) deviceProp.totalGlobalMem);
    printf(" GPU Clock rate: %.0f MHz (%0.2f GHz)\n",
    deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
    printf(" Memory Clock rate: %.0f Mhz\n",
    deviceProp.memoryClockRate * 1e-3f);
    printf(" Memory Bus Width: %d-bit\n",
    deviceProp.memoryBusWidth);
    if (deviceProp.l2CacheSize) {
        printf(" L2 Cache Size: %d bytes\n",
        deviceProp.l2CacheSize);
    }
    printf(" Max Texture Dimension Size (x,y,z) 1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n",
            deviceProp.maxTexture1D,
            deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
            deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);

    printf(" Max Layered Texture Size (dim) x layers 1D=(%d) x %d, 2D=(%d,%d) x %d\n",
        deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1],
        deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
        deviceProp.maxTexture2DLayered[2]);

    printf(" Total amount of constant memory: %lu bytes\n", deviceProp.totalConstMem);
    printf(" Total amount of shared memory per block: %lu bytes\n", deviceProp.sharedMemPerBlock);
    printf(" Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
    printf(" Warp size: %d\n", deviceProp.warpSize);
    printf(" Maximum number of threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf(" Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf(" Maximum sizes of each dimension of a block: %d x %d x %d\n",
        deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf(" Maximum sizes of each dimension of a grid: %d x %d x %d\n",
        deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf(" Maximum memory pitch: %lu bytes\n", deviceProp.memPitch);
    exit(EXIT_SUCCESS);
}

__global__ void matrix_create_gpu(float* matrix, int height, int width)
{
    int row = blockIdx.x + threadIdx.x;

    for(int i = 0; i < width; i++) {
        matrix[row * width +i] = (float)(((rand() % 0xFF) - 128) * 1.0 / 256);
    }

    return;
}

//在Host中申请一块GPU内存，并且使用GPU来给随机初始化, batch_num >= 32, 一个线程做一行，一个block最少做32行
float* matrix_create(int batch_num, int height, int width)
{
    float *matrix;
    cudaMalloc((void **)(&matrix), batch_num * height * width * sizeof(float));
    if (!matrix)
        return NULL;

    dim3 block(32, 1);
    dim3 grid(batch_num * height / 32, 1);

    matrix_create_gpu<<<grid, block>>>(matrix, height, width);
    return matrix;
}

void matrix_delete(void* matrix)
{
    cudaFree(matrix);
    return;
}

__global__ void matrix_add_gpu(int batch_num, float* matrix_a, float* matrix_b, float* matrix_c, char* mask, int height, int width)
{
    //32个线程做一行
    int thread_len = width / blockDim.x;
    int l_pos = threadIdx.y * width + threadIdx.x * thread_len;
    int g_pos = blockIdx.x * height * width + l_pos;

    for(int i = 0 i < thread_len; i++)
        if (mask && mask[l_pos])
            c[g_pos +i] = a[g_pos +i] + b[g_pos +i];
        else
            c[g_pos +i] = 0;

    return;
}

void matrix_add(int batch_num, float* matrix_a, float* matrix_b, float* matrix_c, char* mask, int height, int width)
{
    dim3 block(32, height, 1);
    dim3 grid(batch_num, 1, 1);

    matrix_add_gpu<<<grid, block>>>(matrix_a, matrix_b, matrix_c, mask, int height, int width);

    return;
}

__global__ void matrix_multi_gpu(int batch_num, float *matrix_a, float *matrix_b, float *matrix_c, int height_a,
                                    int width_a, int height_b,
                                    int width_b, int height_c,
                                    int width_c)
{
    #define LOCAL_BLOCK_SIZE 32
    __shared__ float ds_M[LOCAL_BLOCK_SIZE][LOCAL_BLOCK_SIZE];
    __shared__ float ds_N[LOCAL_BLOCK_SIZE][LOCAL_BLOCK_SIZE];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int Row = by * LOCAL_BLOCK_SIZE + ty;
    int Col = bx * LOCAL_BLOCK_SIZE + tx;
    float Pvalue = 0;
    for (int m = 0; m < (width_a - 1) / LOCAL_BLOCK_SIZE + 1; ++m) {
        if (Row < height_a && m * LOCAL_BLOCK_SIZE + tx < width_a) {
            ds_M[ty][tx] = matrix_a[Row * width_a + m * LOCAL_BLOCK_SIZE + tx];
        } else {
            ds_M[ty][tx] = 0.0;
        }

        if (Col < width_b && m * LOCAL_BLOCK_SIZE + ty < height_b) {
            ds_N[ty][tx] = matrix_b[(m * LOCAL_BLOCK_SIZE + ty) * width_b + Col];
        } else {
            ds_N[ty][tx] = 0.0;
        }
        __syncthreads();

        for (int k = 0; k < LOCAL_BLOCK_SIZE; ++k) {
            Pvalue += ds_M[ty][k] * ds_N[k][tx];
        }
        __syncthreads();
    }

    if (Row < height_c && Col < width_c) {
        matrix_c[Row * width_c + Col] = Pvalue;
    }
    return;
}

void matrix_multi(int batch_num, float* matrix_a, int a_height, int a_width, float* matrix_b, int b_width, float* matrix_c)
{
    dim3 block(32, 32, 1);
    dim3 grid(a_height/32, a_width/32, batch_num);

    matrix_multi_gpu<<<grid, block>>>(batch_num1, matrix_a, matrix_b, matrix_c, a_height, a_width, int a_width, b_width, a_height, b_width);
    return;
}

__global__ void matrix_multi_without_transpose_gpu(int batch_num, float *matrix_a, float *matrix_b, float *matrix_c, int height_a,
                                    int width_a, int height_b,
                                    int width_b, int height_c,
                                    int width_c, char* mask, double scale)
{
    #define LOCAL_BLOCK_SIZE 32
    __shared__ float ds_M[LOCAL_BLOCK_SIZE][LOCAL_BLOCK_SIZE];
    __shared__ float ds_N[LOCAL_BLOCK_SIZE][LOCAL_BLOCK_SIZE];
    int g_pos_a = blockIdx.z * height_a * width_a;
    int g_pos_b = blockIdx.z * height_b * width_b;
    int g_pos_c = = blockIdx.z * height_c * width_b;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int Row = by * LOCAL_BLOCK_SIZE + ty;
    int Col = bx * LOCAL_BLOCK_SIZE + tx;
    float Pvalue = 0;
    for (int m = 0; m < (width_a - 1) / LOCAL_BLOCK_SIZE + 1; ++m) {
        if (Row < height_a && m * LOCAL_BLOCK_SIZE + tx < width_a) {
            ds_M[ty][tx] = matrix_a[g_pos_a + Row * width_a + m * LOCAL_BLOCK_SIZE + tx];
        } else {
            ds_M[ty][tx] = 0.0;
        }

        if (Col < height_b && m * LOCAL_BLOCK_SIZE + tx < height_b) {
            ds_N[ty][tx] = matrix_b[g_pos_b + (m * LOCAL_BLOCK_SIZE + ty) * width_b + Col];
        } else {
            ds_N[ty][tx] = 0.0;
        }
        __syncthreads();

        for (int k = 0; k < LOCAL_BLOCK_SIZE; ++k) {
            Pvalue += ds_M[ty][k] * ds_N[k][tx];
        }
        __syncthreads();
    }

    if (Row < height_c && Col < width_c) {
        matrix_c[g_pos_c + Row * width_c + Col] = Pvalue/scale;
    }
    return;
}

void matrix_multi_without_transpose(int batch_num, float* matrix_a, int a_height, int a_num, int a_width,
                                                       float* matrix_b, int b_height, int b_width,
                                                       char* mask, double scale, float* matrix_c)
{
    dim3 block(32, 32, 1);
    dim3 grid(a_height/32, a_width/32, batch_num);

    matrix_multi_without_transpose_gpu<<<grid, block>>>(batch_num1, matrix_a, matrix_b, matrix_c, a_height, a_width, int a_width, b_width, a_height, b_width, mask, scale);
    return;
}

__global__ void add_layer_norm_gpu(int batch_num, float* matrix_a, int height, int width, float* matrix_c, char* gamme, char* beta)
{
    int row = grid.z * height * width + threadIdx.x * width;
    //#0 一个线程做一行
    //mean
    for (int x = 0; x < width; x++) {
        mean += matrix_a[row + x];
    }
    mean = mean / width;

    //var
    for (int x = 0; x < width; x++) {
        tmp = matrix_a[row + x] - mean;
        tmp = tmp * tmp;
        var += tmp;
    }
    var = var / width;
    var += 1e-3;//avoid to too small
    var = sqrt(var);

    for (int x = 0; x < width; x++) {
        tmp = (matrix_a[row + x] - mean) / var;
        matrix_c[row + x] = (float)(gamme[x] * tmp + beta[x]);
    }

}

void add_layer_norm(int batch_num, float* matrix_a, int height, int width, float* matrix_c, char* gamme, char* beta)
{
    dim3 block(width, 1, 1);
    dim3 grid(1, 1, batch_num);
    add_layer_norm_gpu<<<grid, block>>>(int batch_num, float* matrix_a, int height, int width, float* matrix_c, char* gamme, char* beta);
    return;
}

__global__ void softmax_gpu(int batch_num, float* matrix_a, int height, int width, float* matrix_c)
{
    int row = grid.z * height * width + threadIdx.x * width;
    double max_value = 0.0, sum = 0.0;

    for (int x = 0; x < width; x++) {
        if (max_value < matrix_a[row + x])
            max_value = matrix_a[row + x];
    }
    max_value -= 1.0;

    for (int x = 0; x < width; x++) {
        sum += exp(matrix_a[row + x] - max_value);
    }

    for (int x = 0; x < width; x++) {
        matrix_c[row + x] = (float)(exp(matrix_a[row + x] - max_value) / sum);
    }
    return;
}

void softmax(int batch_num, float* matrix_a, int height, int width, float* matrix_c)
{
    dim3 block(width, 1, 1);
    dim3 grid(1, 1, batch_num);
    softmax_gpu<<<grid, block>>>(batch_num, matrix_a, height, width, matrix_c, gamme, beta);

    return;
}