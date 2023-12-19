#include <math.h>
#include <vector>
#include "../kernal/kernal_cpu.h"
#include "ffn_layer.h"

FfnLayer::FfnLayer(int in_dim, int out_dim)
{
    createMatrix(in_dim, out_dim);

    height_ = in_dim;
    width_ = out_dim;
}

FfnLayer::~FfnLayer()
{

}

void FfnLayer::createMatrix(int in_dim, int out_dim)
{
    matrix_ = matrix_create_cpu(in_dim, out_dim);
    return;
}

bool FfnLayer::forward(ForwardData* in, ForwardData* out)
{
    if (in->width_ != height_) {
        printf("ERROR the FfnLayer::forward error, the shape is error, height:%d, ffn width:%d\n",
            in->width_, height_);
        return false;
    }

    //#1 调用GPU或者CPU来进行矩阵的相乘
    matrix_multi_cpu(in->matrix_, matrix_, out->matrix_, in->height_, in->width_, width_);

    out->height_ = in->height_;
    out->width_ = width_;
    return true;
}
