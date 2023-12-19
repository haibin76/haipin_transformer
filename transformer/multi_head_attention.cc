#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../kernal/kernal_cpu.h"
#include "multi_head_attention.h"

MultiHeadAttention::MultiHeadAttention(int max_num, int q_word_dim, int k_word_dim, int v_word_dim)
{
    if (q_word_dim != 512)
        printf("ERROR: the word dimension:%d is not 512!\n", q_word_dim);
    if ((q_word_dim <= 0) || (q_word_dim & 31)) {
        printf("WARNNING: the word dimension:%d is not the 32 multiple\n", q_word_dim);
    }

    max_words_ = max_num;
    q_word_dim_ = q_word_dim;
    k_word_dim_ = k_word_dim;
    v_word_dim_ = v_word_dim;

    newMemoryForVariable();

    return;
}

MultiHeadAttention::~MultiHeadAttention()
{
    for (int i = 0; i < 8; i++) {
        if (q_[i])
            delete[] q_[i];
        if (k_[i])
            delete[] k_[i];
        if (v_[i])
            delete[] v_[i];
        if (r_[i])
            delete[] r_[i];
    }

    delete q_liner_;
    delete k_liner_;
    delete v_liner_;
    delete out_liner_;
    return;
}

void MultiHeadAttention::forward(ForwardData* q, ForwardData* k, ForwardData*v, ForwardData* out, char* mask)
{
    //#0 first input ffn_layer
    q_liner_->forward(q, q);
    q_liner_->forward(k, k);
    q_liner_->forward(v, v);

    //#1 splite q k v into 8 sdpa
    //#2 split q k v into q_ k_ v_, this step can been optimzed, but, not
    int q_num = q_word_dim_ / 8, k_num = k_word_dim_ / 8, v_num = v_word_dim_ / 8;

    for (int y = 0; y < q->height_; y++) {
        for (int x = 0; x < 8; x++) {
            memcpy(&q_[x][y * q_num], &q->matrix_[y * q->width_ + x * q_num], q_num * sizeof(float));
            memcpy(&k_[x][y * q_num], &k->matrix_[y * k->width_ + x * k_num], k_num * sizeof(float));
            memcpy(&v_[x][y * q_num], &v->matrix_[y * v->width_ + x * v_num], v_num * sizeof(float));
        }
    }

    int sdpa_height, sdpa_width;
    for (int i = 0; i < 8; i++) {
        scale_dot_product_attention(q_[i], q->height_, q_num,
                                    k_[i], k->height_, k_num,
                                    v_[i], v->height_, v_num,
                                    mask, q_num,
                                    r_[i], &sdpa_height, &sdpa_width);
        //concat the v
        for (int y = 0; y < q->height_; y++)
            memcpy(&out->matrix_[y * v_word_dim_ + i * v_num], &r_[i], v_num * sizeof(float));
    }

    //#3 finally, input the out_liner
    out->height_ = q->height_;
    out->width_ = q->width_;
    out_liner_->forward(out, out);

    return;
}

void MultiHeadAttention::newMemoryForVariable()
{
    int q_num = q_word_dim_ / 8;
    int k_num = k_word_dim_ / 8;
    int v_num = v_word_dim_ / 8;

    for (int i = 0; i < 8; i++) {
        q_[i] = new float[max_words_ * q_num];
        k_[i] = new float[max_words_ * k_num];
        v_[i] = new float[max_words_ * v_num];

        r_[i] = new float[max_words_ * v_num];
    }

    q_liner_ = new FfnLayer(q_word_dim_, q_word_dim_);
    k_liner_ = new FfnLayer(k_word_dim_, k_word_dim_);
    v_liner_ = new FfnLayer(v_word_dim_, v_word_dim_);

    out_liner_ = new FfnLayer(v_word_dim_, v_word_dim_);

    return;
}