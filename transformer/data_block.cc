#include <string.h>
#include "data_block.h"
#include "../kernal/kernal.h"

void new_fd(ForwardData* in_fd, int batch_num, int height, int width)
{
    in_fd->matrix_ = matrix_create(batch_num, height, width);
    in_fd->batch_num_ = batch_num;
    in_fd->height_ = height;
    in_fd->width_ = width;

    return;
}

void fd_assignment(ForwardData* in_fd, ForwardData* out_fd)
{
    out_fd->matrix_ = matrix_create(in_fd->batch_num_, in_fd->height_, in_fd->width_);

    memcpy(out_fd->matrix_, in_fd->matrix_, in_fd->batch_num_ * in_fd->height_ * in_fd->width_ * sizeof(float));

    out_fd->batch_num_ = in_fd->batch_num_;
    out_fd->height_ = in_fd->height_;
    out_fd->width_ = in_fd->width_;

    return;
}

void delete_fd(ForwardData* in_fd)
{
    delete[] in_fd->matrix_;
    return;
}
