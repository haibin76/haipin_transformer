#ifndef __MULTI_HEAD_ATTENTION_H__
#define __MULTI_HEAD_ATTENTION_H__

#include "haipin_define.h"
#include "ffn_layer.h"
#include "attention.h"

class MultiHeadAttention
{
public:
    MultiHeadAttention(int batch_dim, int sentence_dim, int word_dim);
    ~MultiHeadAttention();
    void forward(ForwardData* q_fd, ForwardData* k_fd, ForwardData* v_fd, ForwardData* mask, ForwardData* out_fd);

private:
    void createFrameWork(int batch_dim, int gu_sentence_dim, int word_dim);

private:
    FfnLayer* q_liner_;
    FfnLayer* k_liner_;
    FfnLayer* v_liner_;
    Attention* attention_[8];

    FfnLayer* out_liner_;
};

#endif
