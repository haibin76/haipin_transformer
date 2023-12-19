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


//����һ����������w��b���ǻ���ƽ���ֲ���������ɵ�
HaipinMatrix* matrix_create(int height, int width);

//����ļӷ�����ָ����������ֵ��ӣ����ֵ�Ǳ�w�����棬��b��û��ϵ��,
//���a��b����״��һ������false�����򷵻�true
bool    matrix_add(HaipinMatrix *a, HaipinMatrix *b, HaipinMatrix *c);

#endif
