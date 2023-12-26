#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../kernal/kernal_cpu.h"
#include "multi_head_attention.h"

MultiHeadAttention::MultiHeadAttention(int batch_dim, int sentence_dim, int word_dim)
{
    if (word_dim != 512)
        printf("ERROR: the word dimension:%d is not 512!\n", word_dim);
    if ((word_dim <= 0) || (word_dim & 31)) {
        printf("WARNNING: the word dimension:%d is not the 32 multiple\n", word_dim);
    }

    createFrameWork(batch_dim, sentence_dim, word_dim);

    return;
}

MultiHeadAttention::~MultiHeadAttention()
{
    for (int i = 0; i < 8; i++)
        delete[] attention_;

    delete q_liner_;
    delete k_liner_;
    delete v_liner_;
    delete out_liner_;

    return;
}

void MultiHeadAttention::forward(ForwardData* q_fd, ForwardData* k_fd, ForwardData* v_fd, ForwardData* mask, ForwardData* out_fd)
{
    int head_num = 8;
    int num = q_fd->width_ / head_num;
    //q_fd->width_ = k_fd->width_ = v_fd->width_, but, now, we dont check this

    //#0 input the linear
    q_liner_->forward(q_fd, q_fd);
    k_liner_->forward(k_fd, k_fd);
    v_liner_->forward(v_fd, v_fd);

    //#1 input the 8 multi head attention
    ForwardData q = *q_fd, k = *k_fd, v = *v_fd, m = *mask, o = *out_fd;
    q.num_ = k.num_ = v.num_ = num;
    for (int i = 0; i < head_num; i++) {
        q.matrix_ = q_fd->matrix_ + i * num;
        k.matrix_ = k_fd->matrix_ + i * num;
        v.matrix_ = v_fd->matrix_ + i * num;
        o.matrix_ = out_fd->matrix_ + i * num;

        attention_[i]->forward(&q, &k, &v, mask, &o);
    }

    //#2 input the out_liner
    out_liner_->forward(out_fd, out_fd);

    return;
}

void MultiHeadAttention::createFrameWork(int batch_dim, int sentence_dim, int word_dim)
{
    //dont check success or failed! because busy
    q_liner_ = new FfnLayer(word_dim, word_dim);
    k_liner_ = new FfnLayer(word_dim, word_dim);
    v_liner_ = new FfnLayer(word_dim, word_dim);

    //#malloc the 8 sdpa
    for (int i = 0; i < 8; i++)
        attention_[i] = new Attention(batch_dim, sentence_dim, word_dim);

    out_liner_ = new FfnLayer(word_dim, word_dim);

    return;
}