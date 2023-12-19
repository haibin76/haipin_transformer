#include <math.h>
#include <vector>
#include "../kernal/kernal_cpu.h"
#include "softmax.h"

SoftMax::SoftMax(int dim)
{
    dim_ = dim;

    buffer_ = new double[dim_];

    return;
}

SoftMax::~SoftMax()
{
    if (buffer_)
        delete[] buffer_;

    return;
}

void SoftMax::forward(ForwardData* in, ForwardData* out)
{
    softmax_cpu(in->matrix_, out->matrix_, buffer_, in->height_, in->width_);

    out->height_ = in->height_;
    out->width_ = in->width_;
    return;
}
