#include <math.h>
#include <vector>
#include "transformer.h"

using namespace std;
Transformer::Transformer(int max_word, int word_dim)
{
    if (word_dim != 512)
        printf("WARNNING: the word dimension:%d is not 512!\n", word_dim);
    if ((word_dim <= 0) || (word_dim & 31)) {
        printf("WARNNING: the word dimension:%d is not the 32 multiple\n", word_dim);
    }

    max_word_ = max_word;
    word_dim_ = word_dim;

    //申请一块buffer，用于保存PositionEmbeding
    pe_ = new float [max_word * word_dim];

    //计算PositionEmbeding
    calcPositionEmbeding();

    return;
}

Transformer::~Transformer()
{
    if (pe_)
        delete [] pe_;
}

void Transformer::forward(std::vector<float*>& sentence)
{
    int word_pos = 0;
    for (std::vector<float*>::iterator it = sentence.begin(); it != sentence.end(); it++, word_pos++) {
        float* word_vector = *it;
        for(int i = 0; i < word_dim_; i+=2) {
            word_vector[i] += pe_[word_pos * word_dim_ + i];
            word_vector[i + 1] += pe_[word_pos * word_dim_ + i + 1];
        }
    }

    return;
}

void Transformer::calcPositionEmbeding()
{
    //论文中可能会溢出，这里要做一次变换
    //1 / (10000 ^ 2i/d_model) = 10000 ^ -(2i/d_model) = exp(ln(10000 ^ -(2i/d_model)))
    // = exp(-(2i/d_model) * ln(10000))
    for(int pos = 0; pos < max_word_; pos++)
        for(int i = 0; i < word_dim_; i+= 2) {
            double tmp = -1 * (i * 1.0 / word_dim_) * log(10000);
            float tmp2 = (float)(exp(tmp));
            pe_[pos * word_dim_ + i] = (float)sin (pos * tmp2);
            pe_[pos * word_dim_ + i + 1] = (float)cos(pos * tmp2);
    }

    return;
}