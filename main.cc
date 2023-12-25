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
void create_sentence(ForwardData* sentence)
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

    int word_dim = 512, in_vocab_size = 4096, genuine_vocab_size = 20000;
    int in_max_words = 16, genuine_max_words = 32;
    ForwardData in_sentence, genuine_sentence, out_sentence;
    in_sentence.height_ = 4;
    in_sentence.width_ = word_dim;
    genuine_sentence.height_ = 5;
    genuine_sentence.width_ = word_dim;
    create_sentence(&in_sentence);
    create_sentence(&genuine_sentence);

    PositionEmbeding* in_pe = new PositionEmbeding(in_max_words, word_dim);
    in_pe->forward(&in_sentence, &in_sentence);
    PositionEmbeding* genuine_pe = new PositionEmbeding(genuine_max_words, word_dim);
    genuine_pe->forward(&genuine_sentence, &genuine_sentence);

    Transformer* tf = new Transformer(in_max_words, word_dim, genuine_max_words, in_vocab_size, genuine_vocab_size);
    tf->forward(&in_sentence, &genuine_sentence, &out_sentence);

    return 0;
}