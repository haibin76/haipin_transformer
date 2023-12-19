#ifndef __MATRIX_MATH__
#define __MATRIX_MATH__

#include <vector>

typedef struct tagHaipinMatrix
{
    int matrix_height;
    int matrix_width;
    float* w;
    float* b;
}HaipinMatrix;


//生成一个矩阵，其中w和b都是基于平均分布来随机生成的
HaipinMatrix* matrix_create(int height, int width);

//矩阵的加法，是指两个矩阵中值相加，这个值是被w所代替，和b是没关系的,
//如果a和b的形状不一样，则返false，否则返回true
bool    matrix_add(HaipinMatrix *a, HaipinMatrix *b, HaipinMatrix *c);

#endif
