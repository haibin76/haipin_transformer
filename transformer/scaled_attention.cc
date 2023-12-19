#include <math.h>
#include <vector>
#include "scaled_attention.h"

ScaledDotProductAttention::ScaledDotProductAttention(int dim)
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

ScaledDotProductAttention::~ScaledDotProductAttention()
{
    if (pe_)
        delete [] pe_;
}

void ScaledDotProductAttention::forward(std::vector<float*>& sentence)
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
