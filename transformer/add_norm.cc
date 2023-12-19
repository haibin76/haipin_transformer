#include <math.h>
#include <vector>
#include "../kernal/kernal_cpu.h"
#include "add_norm.h"

AddLayerNorm::AddLayerNorm(int dim)
{
    dim_ = dim;

    gamme_ = new char[dim_];
    beta_ = new char[dim_];

    for (int i = 0; i < dim_; i++) {
        gamme_[i] = 1;
        beta_[i] = 0;
    }

    return;
}

AddLayerNorm::~AddLayerNorm()
{
    if (gamme_)
        delete[] gamme_;

    if (beta_)
        delete[] beta_;

    return;
}

void AddLayerNorm::forward(ForwardData* in, ForwardData* out)
{
    matrix_add_cpu(in->matrix_, out->matrix_, in->matrix_, in->height_, in->width_);
    add_layer_norm_cpu(in->matrix_, out->matrix_, gamme_, beta_, in->height_, in->width_);

    out->height_ = in->height_;
    out->width_ = in->width_;
    return;
}
