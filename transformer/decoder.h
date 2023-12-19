#ifndef __DECODER_H__
#define __DECODER_H__

#include "multi_head_attention.h"
#include "add_norm.h"
#include "ffn_layer.h"

class Decoder
{
public:
    Decoder(int max_word, int word_dim);
    ~Decoder();
    void forward(ForwardData* in, ForwardData* kv, ForwardData* out);

private:
    void createFrameWork(int max_word, int word_dim);
    void relu(ForwardData* in, ForwardData* out);
    void setMask(int word_num);

private:
    int max_word_;
    int word_dim_;

    MultiHeadAttention* mh_attention_msk_;
    AddLayerNorm* a_ln0_;

    MultiHeadAttention* mh_attention_;
    AddLayerNorm* a_ln1_;

    FfnLayer* ffn0_;
    FfnLayer* ffn1_;
    AddLayerNorm* a_ln2_;

    float* q_;
    float* k_;
    float* v_;
    float* ffn_hide_;
    char* msk_;
};

#endif
