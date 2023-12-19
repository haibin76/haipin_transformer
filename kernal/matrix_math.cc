#include <math.h>
#include "matrix_math.h"
#include "matrix_cpu.h"

HaipinMatrix* matrix_create(int height, int width)
{
    HaipinMatrix* hm = new HaipinMatrix();
    if (!hm)
        goto _faild;

    hm->w = new float[height * width];
    if (!hm->w)
        goto _faild;

    hm->b = new float[height * width];
    if (!hm->b)
        goto _faild;

    hm->matrix_height = height;
    hm->matrix_width = width;
    return hm;

_faild:
    if (hm) {
        if (hm->b)
            delete[] hm->b;
        if (hm->w)
            delete[] hm->w;
        delete hm;
    }

    return NULL;
}

bool    matrix_add(HaipinMatrix* a, HaipinMatrix* b, HaipinMatrix* c)
{
    if ((a->matrix_height != b->matrix_height) || (a->matrix_width != b->matrix_width)) {
        printf("a shape[%d,%d] != b shape[%d,%d]\n", a->matrix_height, a->matrix_width, b->matrix_width, b->matrix_width);
        return false;
    }

    c->matrix_height = a->matrix_height;
    c->matrix_width = a->matrix_width;
    matrix_add_cpu(a->w, b->w, c->w, a->matrix_height, a->matrix_width);

    return true;
}