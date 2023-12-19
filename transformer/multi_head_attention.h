#ifndef __MULTI_HEAD_ATTENTION_H__
#define __MULTI_HEAD_ATTENTION_H__

#include "haipin_define.h"
#include "ffn_layer.h"

class MultiHeadAttention
{
public:
    MultiHeadAttention(int max_num, int q_word_dim, int k_word_dim, int v_word_dim);
    ~MultiHeadAttention();
    void forward(ForwardData* q, ForwardData* k, ForwardData* v, ForwardData* out, char* mask);

private:
    void newMemoryForVariable();

private:
    int max_words_;
    int q_word_dim_;
    int k_word_dim_;
    int v_word_dim_;

    float* q_[8];
    float* k_[8];
    float* v_[8];
    float* r_[8];

    FfnLayer* q_liner_;
    FfnLayer* k_liner_;
    FfnLayer* v_liner_;
    FfnLayer* out_liner_;
};

#endif
