#include <math.h>
#include <vector>
#include "../kernal/kernal.h"
#include "softmax.h"

SoftMax::SoftMax(int dim)
{
    dim_ = dim;

    return;
}

SoftMax::~SoftMax()
{
    return;
}

void SoftMax::forward(ForwardData* in_fd, ForwardData* out_fd)
{
    softmax(in_fd->batch_num_, in_fd->matrix_, in_fd->height_, in_fd->width_, out_fd->matrix_);

    out_fd->batch_num_ = in_fd->batch_num_;
    out_fd->height_ = in_fd->height_;
    out_fd->width_ = in_fd->width_;
    return;
}
