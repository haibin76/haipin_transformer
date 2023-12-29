#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../kernal/kernal.h"
#include "attention.h"

Attention::Attention(int batch_dim, int sentence_dim, int word_dim)
{
    if (word_dim != 512)
        printf("ERROR: the word dimension:%d is not 512!\n", word_dim);
    if ((word_dim <= 0) || (word_dim & 31)) {
        printf("WARNNING: the word dimension:%d is not the 32 multiple\n", word_dim);
    }

    createFrameWork(batch_dim, sentence_dim, word_dim);

    return;
}

Attention::~Attention()
{
    matrix_delete(fd_.matrix_);
    return;
}

void Attention::forward(ForwardData* q_fd, ForwardData* k_fd, ForwardData* v_fd, ForwardData* mask_fd, ForwardData* out_fd)
{
    //k_fd->height_ == q_fd->height_ == v_fd->height
    //the batch_nums in the q_fd|k_fd|v_fd is the same, we omit to check, because busy
    //#0 q * k ^t / scale
    matrix_multi_without_transpose(q_fd->batch_num_, q_fd->matrix_, q_fd->height_, q_fd->num_, q_fd->width_,
                                        k_fd->matrix_, k_fd->height_, k_fd->width_,
                                        (char*)mask_fd->matrix_, sqrt(q_fd->num_), fd_.matrix_);
    fd_.batch_num_ = q_fd->batch_num_;
    fd_.height_ = q_fd->height_;
    fd_.width_ = k_fd->height_;

    //#1 softmax(q * k ^t /scale)
    softmax(fd_.batch_num_, fd_.matrix_, fd_.height_, fd_.width_, fd_.matrix_);

    //#2 the scores * v_fd
    matrix_multi(fd_.batch_num_, fd_.matrix_, fd_.height_, fd_.width_, v_fd->matrix_, v_fd->width_, out_fd->matrix_);
    out_fd->batch_num_ = fd_.batch_num_;
    out_fd->height_ = fd_.height_;
    out_fd->width_ = v_fd->width_;

    return;
}

void Attention::createFrameWork(int batch_dim, int sentence_dim, int word_dim)
{
    fd_.batch_num_  = batch_dim;
    fd_.height_ = sentence_dim;
    fd_.width_ = sentence_dim;

    fd_.matrix_ = matrix_create(fd_.batch_num_, fd_.height_, fd_.width_);
    return;
}