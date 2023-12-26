#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "position_embeding.h"
#include "transformer/transformer.h"
#include "transformer/haipin_define.h"
#include "transformer/add_norm.h"
#include "transformer/haipin_define.h"

//模拟python层的一些函数，最终这些都要被python替换掉

//使用code取随机生成一个句子的词向量
void create_fd(ForwardData* sentence)
{
    sentence->matrix_ = new float[sentence->height_ * sentence->width_];

    for (int i = 0; i < sentence->height_ * sentence->width_; i++) {
        sentence->matrix_[i] = (float)(((rand() % 256) - 128) / 128.0);
    }

    return;
}

void release_sentence(ForwardData* sentence)
{
    delete[] sentence->matrix_;
    return;
}

int main(char argc, char** argv)
{
    /*int word_dim = 512, word_num_in_sentence = 128;
    PositionEmbeding* pe = new PositionEmbeding(word_num_in_sentence, word_dim);

    std::vector<float*> sentence;
    create_sentence(sentence, word_num_in_sentence, word_dim);

    pe->forward(sentence);

    release_sentence(sentence);
    delete pe;
    float in_matrix[12] = { 7.2516, 5.5347, 2.0426, 7.5508, 7.8015, 3.3618, 3.4879, 6.4329, 3.8357, 6.2470, 1.5781, 0.9355};
    float out_matrix[12] = { 7.2516, 5.5347, 2.0426, 7.5508, 7.8015, 3.3618, 3.4879, 6.4329, 3.8357, 6.2470, 1.5781, 0.9355 };
    ForwardData  in, out;
    in.height_ = 3;
    in.width_ = 4;
    in.matrix_ = &in_matrix[0];

    out.matrix_ = &out_matrix[0];
    AddLayerNorm* an = new AddLayerNorm(4);
    an->forward(&in, &out);*/

    //set some constant parameters
    int word_dim = 512, batch_size = 64;
    int in_sentence_dim = 32, in_vocab_size = 4096;
    int gu_sentence_dim = 48, gu_vocab_size = 8192;

    //create input & output data, and initiate some variables
    ForwardData in_data, gu_data, out_data;
    in_data.height_ = in_sentence_dim;
    in_data.width_ = word_dim;
    in_data.batch_num_ = batch_size;
    create_fd(&in_data);

    gu_data.height_ = gu_sentence_dim;
    gu_data.width_ = word_dim;
    gu_data.batch_num_ = batch_size;
    create_fd(&gu_data);

    out_data.height_ = gu_sentence_dim;
    out_data.width_ = word_dim;
    out_data.batch_num_ = batch_size;
    create_fd(&out_data);

    //malloc the position of pe for input
    PositionEmbeding* in_pe = new PositionEmbeding(in_sentence_dim, word_dim);
    in_pe->forward(&in_data, &in_data);

    //malloc the positio of pe for gu
    PositionEmbeding* gu_pe = new PositionEmbeding(gu_sentence_dim, word_dim);
    gu_pe->forward(&gu_data, &gu_data);

    //start to transformer
    Transformer* tf = new Transformer(batch_size, word_dim, in_sentence_dim, gu_sentence_dim, in_vocab_size, gu_vocab_size);
    tf->forward(&in_data, &gu_data, &out_data);

    return 0;
}