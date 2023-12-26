#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "data_block.h"
#include "encoder.h"

Encoder::Encoder(int batch_dim, int in_sentence_dim, int word_dim)
{
    if (word_dim != 512)
        printf("WARNNING: the word dimension:%d is not 512!\n", word_dim);
    if ((word_dim <= 0) || (word_dim & 31)) {
        printf("WARNNING: the word dimension:%d is not the 32 multiple\n", word_dim);
    }

    //计算PositionEmbeding
    createFrameWork(batch_dim, in_sentence_dim, word_dim);

    return;
}

Encoder::~Encoder()
{
    delete mh_attention_;
    delete a_ln0_;

    delete ffn0_;
    delete ffn1_;
    delete a_ln1_;

    delete_fd(&q_fd_);
    delete_fd(&k_fd_);
    delete_fd(&v_fd_);

    return;
}

void Encoder::forward(ForwardData* in_fd, ForwardData* out_fd)
{
    //input the multi_head_attention
    fd_assignment(in_fd, &q_fd_);
    fd_assignment(in_fd, &k_fd_);
    fd_assignment(in_fd, &v_fd_);

    mh_attention_->forward(&q_fd_, &k_fd_, &v_fd_, out_fd, NULL);
    a_ln0_->forward(in_fd, out_fd);

    //进入前反馈链接层
    fd_assignment(out_fd, &q_fd_);
    ffn0_->forward(&q_fd_, &m_fd_);
    relu(&m_fd_, &m_fd_);
    ffn1_->forward(&m_fd_, in_fd);

    a_ln1_->forward(in_fd, out_fd);

    return;
}

void Encoder::createFrameWork(int batch_dim, int in_sentence_dim, int word_dim)
{
    //申请必要的资源，不做申请成功与不成功的检测
    mh_attention_ = new MultiHeadAttention(batch_dim, in_sentence_dim, word_dim);
    a_ln0_ = new AddLayerNorm(word_dim);

    ffn0_ = new FfnLayer(word_dim, 2048);
    ffn1_ = new FfnLayer(2048, word_dim);
    a_ln1_ = new AddLayerNorm(word_dim);

    new_fd(&q_fd_, batch_dim, in_sentence_dim, word_dim);
    new_fd(&k_fd_, batch_dim, in_sentence_dim, word_dim);
    new_fd(&v_fd_, batch_dim, in_sentence_dim, word_dim);
    new_fd(&m_fd_, 1, 2048, word_dim);
    return;
}

void Encoder::relu(ForwardData* in_fd, ForwardData* out_fd)
{
    for (int b = 0; b < in_fd->batch_num_; b++) {
        int batch_offset = b * in_fd->height_ * in_fd->width_;
        for (int y = 0; y < in_fd->height_; y++)
            for (int x = 0; x < in_fd->width_; x++) {
                if (in_fd->matrix_[batch_offset + y * in_fd->width_ + x] < 0)
                    in_fd->matrix_[batch_offset + y * in_fd->width_ + x] = 0.0;
            }
    }

    out_fd->height_ = in_fd->height_;
    out_fd->width_ = in_fd->width_;

    return;
}