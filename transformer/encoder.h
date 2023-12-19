#ifndef __ENCODER_H__
#define __ENCODER_H__

#include "multi_head_attention.h"
#include "add_norm.h"
#include "ffn_layer.h"

class Encoder
{
public:
    Encoder(int max_word, int word_dim);
    ~Encoder();
    void forward(ForwardData* in, ForwardData* out);

private:
    void createFrameWork(int max_word, int word_dim);
    void relu(ForwardData* in, ForwardData* out);

private:
    int max_word_;
    int word_dim_;

    MultiHeadAttention* mh_attention_;
    AddLayerNorm* a_ln0_;
    FfnLayer* ffn0_;
    FfnLayer* ffn1_;
    AddLayerNorm* a_ln1_;

    float* q_;
    float* k_;
    float* v_;
    float* ffn_hide_;
};

#endif
