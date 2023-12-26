#include <math.h>
#include <vector>
#include "../kernal/kernal_cpu.h"
#include "add_norm.h"

AddLayerNorm::AddLayerNorm(int word_dim)
{
    word_dim_ = word_dim;

    gamme_ = new char[word_dim_];
    beta_ = new char[word_dim_];

    for (int i = 0; i < word_dim_; i++) {
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

void AddLayerNorm::forward(ForwardData* in_fd, ForwardData* out_fd)
{
    matrix_add_cpu(in_fd->batch_num_, in_fd->matrix_, out_fd->matrix_, in_fd->matrix_, NULL, in_fd->height_, in_fd->width_);
    add_layer_norm_cpu(in_fd->batch_num_, in_fd->matrix_, in_fd->height_, in_fd->width_, out_fd->matrix_, gamme_, beta_);

    out_fd->batch_num_ = in_fd->batch_num_;
    out_fd->height_ = in_fd->height_;
    out_fd->width_ = in_fd->width_;
    return;
}
