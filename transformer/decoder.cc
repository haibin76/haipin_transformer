#include <math.h>
#include <vector>
#include "decoder.h"

using namespace std;
Decoder::Decoder(int max_word, int word_dim)
{
    if (word_dim != 512)
        printf("WARNNING: the word dimension:%d is not 512!\n", word_dim);
    if ((word_dim <= 0) || (word_dim & 31)) {
        printf("WARNNING: the word dimension:%d is not the 32 multiple\n", word_dim);
    }

    max_word_ = max_word;
    word_dim_ = word_dim;

    //计算PositionEmbeding
    createFrameWork(max_word, word_dim);

    return;
}

Decoder::~Decoder()
{
    if (q_)
        delete[] q_;
    if (k_)
        delete[] k_;
    if (v_)
        delete[] v_;

    return;
}

void Decoder::forward(ForwardData* in, ForwardData* kv, ForwardData* out)
{
    //input the multi_head_attention, from bottom to up, this is the botton
    ForwardData q = *in, k = *in, v = *in, ffn_h;
    q.matrix_ = q_;
    k.matrix_ = k_;
    v.matrix_ = v_;

    memcpy(q.matrix_, in->matrix_, q.height_ * q.width_ * sizeof(float));
    memcpy(k.matrix_, in->matrix_, k.height_ * k.width_ * sizeof(float));
    memcpy(v.matrix_, in->matrix_, v.height_ * v.width_ * sizeof(float));

    setMask(q.height_);

    mh_attention_msk_->forward(&q, &k, &v, out, msk_);
    a_ln0_->forward(in, out);

    //set the q = out
    in->height_ = out->height_;
    in->width_ = out->width_;
    memcpy(in->matrix_, out->matrix_, in->height_ * in->width_ * sizeof(float));

    q.height_ = out->height_;
    q.width_ = out->width_;
    memcpy(q.matrix_, out->matrix_, q.height_ * q.width_ * sizeof(float));

    k.height_ = kv->height_;
    k.width_ = kv->width_;
    memcpy(k.matrix_, kv->matrix_, k.height_ * k.width_ * sizeof(float));

    v.height_ = kv->height_;
    v.width_ = kv->width_;
    memcpy(v.matrix_, kv->matrix_, v.height_ * v.width_ * sizeof(float));

    mh_attention_msk_->forward(&q, &k, &v, out, msk_);
    a_ln1_->forward(in, out);

    ffn_h.height_ = out->height_;
    ffn_h.width_ = 2048;
    ffn_h.matrix_ = ffn_hide_;
    ffn0_->forward(out, &ffn_h);
    relu(&ffn_h, &ffn_h);
    ffn1_->forward(&ffn_h, in);

    a_ln1_->forward(in, out);

    return;
}

void Decoder::createFrameWork(int max_word, int word_dim)
{
    //dont check, just busy, CAUTION:this max_word is different with encoder max_word
    mh_attention_msk_ = new MultiHeadAttention(max_word, word_dim, word_dim, word_dim);
    a_ln0_ = new AddLayerNorm(word_dim);

    mh_attention_ = new MultiHeadAttention(max_word, word_dim, word_dim, word_dim);
    a_ln1_ = new AddLayerNorm(word_dim);

    ffn0_ = new FfnLayer(word_dim, 2048);
    ffn1_ = new FfnLayer(2048, word_dim);
    a_ln1_ = new AddLayerNorm(word_dim);

    q_ = new float[max_word * word_dim];
    k_ = new float[max_word * word_dim];
    v_ = new float[max_word * word_dim];
    ffn_hide_ = new float[word_dim * 2048];

    msk_ = new char[max_word * max_word * sizeof(char)];
    return;
}

void Decoder::setMask(int word_num)
{
    for (int y = 0; y < word_num; y++)
        for (int x = 0; x < word_num; x++) {
            if (x > y)
                msk_[y * word_num + x] = 0;
            else
                msk_[y * word_num + x] = 1;
        }

    return;
}

void Decoder::relu(ForwardData* in, ForwardData* out)
{
    for (int y = 0; y < in->height_; y++)
        for (int x = 0; x < in->width_; x++) {
            if (in->matrix_[y * in->width_ + x] < 0)
                in->matrix_[y * in->width_ + x] = 0.0;
        }

    out->height_ = in->height_;
    out->width_ = in->width_;

    return;
}