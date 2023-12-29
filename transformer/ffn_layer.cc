#include <math.h>
#include <vector>
#include "../kernal/kernal.h"
#include "ffn_layer.h"

FfnLayer::FfnLayer(int in_dim, int out_dim)
{
    createFrameWork(in_dim, out_dim);
}

FfnLayer::~FfnLayer()
{
    matrix_delete(weight_matrix_);
    matrix_delete(temp_fd_.matrix_);

    return;
}

bool FfnLayer::forward(ForwardData* in_fd, ForwardData* out_fd)
{
    ForwardData o;
    if (in_fd == out_fd) {
        if ((in_fd->batch_num_ * in_fd->height_ * wm_width_) >
            (temp_fd_.batch_num_ * temp_fd_.height_ * temp_fd_.width_)) {
            deleteTempFD();
            createTempFD(in_fd->batch_num_, in_fd->height_, wm_width_);
            o = temp_fd_;
        }
    } else
        o = *out_fd;

    //#1 调用GPU或者CPU来进行矩阵的相乘
    matrix_multi(1, in_fd->matrix_, in_fd->batch_num_ * in_fd->height_, in_fd->width_, weight_matrix_, wm_width_, o.matrix_);
    if (o.matrix_ != out_fd->matrix_)
        memcpy(out_fd->matrix_, o.matrix_, in_fd->batch_num_ * in_fd->height_ * wm_width_ * sizeof(float));

    out_fd->batch_num_ = in_fd->batch_num_;
    out_fd->height_ = in_fd->height_;
    out_fd->width_ = wm_width_;
    return true;
}

void FfnLayer::createFrameWork(int in_dim, int out_dim)
{
    weight_matrix_ = matrix_create(1, in_dim, out_dim);
    wm_height_ = in_dim;
    wm_width_ = out_dim;
    return;
}

void FfnLayer::deleteTempFD()
{
    matrix_delete(temp_fd_.matrix_);
    memset(&temp_fd_, 0, sizeof(ForwardData));

    return;
}

void FfnLayer::createTempFD(int batch_num, int height, int width)
{
    matrix_create(batch_num, height, width);
    temp_fd_.batch_num_ = batch_num;
    temp_fd_.height_ = height;
    temp_fd_.width_ = width;

    return;
}
