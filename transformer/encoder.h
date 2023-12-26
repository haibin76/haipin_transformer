#ifndef __ENCODER_H__
#define __ENCODER_H__

#include "haipin_define.h"
#include "data_block.h"
#include "multi_head_attention.h"
#include "add_norm.h"
#include "ffn_layer.h"

class Encoder
{
public:
    Encoder(int batch_dim, int in_sentence_dim, int word_dim);
    ~Encoder();
    void forward(ForwardData* in_fd, ForwardData* out_fd);

private:
    void createFrameWork(int batch_dim, int in_sentence_dim, int word_dim);
    void relu(ForwardData* in_fd, ForwardData* out_fd);

private:
    int max_word_;
    int word_dim_;

    MultiHeadAttention*     mh_attention_;
    AddLayerNorm*           a_ln0_;

    FfnLayer*               ffn0_;
    FfnLayer*               ffn1_;
    AddLayerNorm*           a_ln1_;

    ForwardData             q_fd_;
    ForwardData             k_fd_;
    ForwardData             v_fd_;
    ForwardData             m_fd_;
};

#endif
