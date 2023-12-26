#ifndef __DECODER_H__
#define __DECODER_H__

#include "multi_head_attention.h"
#include "add_norm.h"
#include "ffn_layer.h"

class Decoder
{
public:
    Decoder(int batch_dim, int gu_sentence_dim, int word_dim);
    ~Decoder();
    void forward(ForwardData* in, ForwardData* kv, ForwardData* out);

private:
    void createFrameWork(int batch_dim, int max_word, int word_dim);
    void relu(ForwardData* in, ForwardData* out);
    void setMask(int sentence_dim);

private:
    MultiHeadAttention* mh_attention_msk_;
    AddLayerNorm* a_ln0_;

    MultiHeadAttention* mh_attention_;
    AddLayerNorm* a_ln1_;

    FfnLayer* ffn0_;
    FfnLayer* ffn1_;
    AddLayerNorm* a_ln2_;

    ForwardData             q_fd_;
    ForwardData             k_fd_;
    ForwardData             v_fd_;
    ForwardData             m_fd_;
    unsigned char*          msk_;
};

#endif
