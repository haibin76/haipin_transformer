#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "position_embeding.h"
#include "transformer/haipin_define.h"
#include "transformer/add_norm.h"

//模拟python层的一些函数，最终这些都要被python替换掉

//使用code取随机生成一个句子的词向量
void create_sentence(std::vector<float*>& sentence, int word_num, int word_dim)
{
    for (int word_idx = 0; word_idx < word_num; word_idx++) {
        float* word_vect = new float[word_dim];
        for (int dim_idx = 0; dim_idx < word_dim; dim_idx++) {
            //随机初始化一个float值，在[0,1]之间
            word_vect[dim_idx] = (float)((rand() % 0xFF) / 256.0);
        }
        sentence.push_back(word_vect);
    }
}

void release_sentence(std::vector<float*>& sentence)
{
    while(!sentence.empty()) {
        std::vector<float*>::iterator it = sentence.begin();
        delete[] *it;
        sentence.erase(it);
    }
}

int main(char argc, char** argv)
{
    /*int word_dim = 512, word_num_in_sentence = 128;
    PositionEmbeding* pe = new PositionEmbeding(word_num_in_sentence, word_dim);

    std::vector<float*> sentence;
    create_sentence(sentence, word_num_in_sentence, word_dim);

    pe->forward(sentence);

    release_sentence(sentence);
    delete pe;*/
    float in_matrix[12] = { 7.2516, 5.5347, 2.0426, 7.5508, 7.8015, 3.3618, 3.4879, 6.4329, 3.8357, 6.2470, 1.5781, 0.9355};
    float out_matrix[12] = { 7.2516, 5.5347, 2.0426, 7.5508, 7.8015, 3.3618, 3.4879, 6.4329, 3.8357, 6.2470, 1.5781, 0.9355 };
    ForwardData  in, out;
    in.height_ = 3;
    in.width_ = 4;
    in.matrix_ = &in_matrix[0];

    out.matrix_ = &out_matrix[0];
    AddLayerNorm* an = new AddLayerNorm(4);
    an->forward(&in, &out);
    return 0;
}