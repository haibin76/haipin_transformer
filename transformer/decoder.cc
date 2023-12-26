#include <math.h>
#include <stdio.h>
#include "data_block.h"
#include "decoder.h"

using namespace std;
Decoder::Decoder(int batch_dim, int gu_sentence_dim, int word_dim)
{
    if (word_dim != 512)
        printf("WARNNING: the word dimension:%d is not 512!\n", word_dim);
    if ((word_dim <= 0) || (word_dim & 31)) {
        printf("WARNNING: the word dimension:%d is not the 32 multiple\n", word_dim);
    }

    //计算PositionEmbeding
    createFrameWork(batch_dim, gu_sentence_dim, word_dim);
    setMask(gu_sentence_dim);

    return;
}

Decoder::~Decoder()
{
    delete mh_attention_msk_;
    delete a_ln0_;

    delete mh_attention_;
    delete a_ln1_;

    delete ffn0_;
    delete ffn1_;
    delete a_ln2_;

    delete_fd(&q_fd_);
    delete_fd(&k_fd_);
    delete_fd(&v_fd_);
    delete_fd(&m_fd_);

    delete []msk_;

    return;
}

void Decoder::forward(ForwardData* in_fd, ForwardData* kv_fd, ForwardData* out_fd)
{
    //input the multi_head_attention, from bottom to up, this is the botton
    //input the multi_head_attention
    ForwardData msk_fd;
    fd_assignment(in_fd, &q_fd_);
    fd_assignment(in_fd, &k_fd_);
    fd_assignment(in_fd, &v_fd_);

    setMask(in_fd->height_);
    msk_fd.batch_num_ = 1;
    msk_fd.height_ = in_fd->height_;
    msk_fd.width_ = in_fd->height_;

    mh_attention_msk_->forward(&q_fd_, &k_fd_, &v_fd_, &msk_fd, out_fd);
    a_ln0_->forward(out_fd, &q_fd_);

    fd_assignment(kv_fd, &k_fd_);
    fd_assignment(kv_fd, &v_fd_);

    mh_attention_msk_->forward(&q_fd_, &k_fd_, &v_fd_, NULL, out_fd);
    a_ln1_->forward(out_fd, in_fd);

    ffn0_->forward(in_fd, &m_fd_);
    relu(&m_fd_, &m_fd_);
    ffn1_->forward(&m_fd_, in_fd);

    a_ln1_->forward(in_fd, out_fd);

    return;
}

void Decoder::createFrameWork(int batch_dim, int gu_sentence_dim, int word_dim)
{
    //int batch_dim, int sentence_dim, int word_dim);
    //dont check, just busy, CAUTION:this max_word is different with encoder max_word
    mh_attention_msk_ = new MultiHeadAttention(batch_dim, gu_sentence_dim, word_dim);
    a_ln0_ = new AddLayerNorm(word_dim);

    mh_attention_ = new MultiHeadAttention(batch_dim, gu_sentence_dim, word_dim);
    a_ln1_ = new AddLayerNorm(word_dim);

    ffn0_ = new FfnLayer(word_dim, 2048);
    ffn1_ = new FfnLayer(2048, word_dim);
    a_ln1_ = new AddLayerNorm(word_dim);

    new_fd(&q_fd_, batch_dim, gu_sentence_dim, word_dim);
    new_fd(&k_fd_, batch_dim, gu_sentence_dim, word_dim);
    new_fd(&v_fd_, batch_dim, gu_sentence_dim, word_dim);
    new_fd(&m_fd_, 1, 2048, word_dim);

    msk_ = new unsigned char[gu_sentence_dim * gu_sentence_dim];

    return;
}

void Decoder::setMask(int sentence_dim)
{
    for (int y = 0; y < sentence_dim; y++)
        for (int x = 0; x < sentence_dim; x++) {
            if (x > y)
                msk_[y * sentence_dim + x] = 0;
            else
                msk_[y * sentence_dim + x] = 1;
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