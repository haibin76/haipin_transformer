#include <math.h>
#include <vector>
#include "position_embeding.h"

using namespace std;
PositionEmbeding::PositionEmbeding(int max_word, int word_dim)
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

PositionEmbeding::~PositionEmbeding()
{
    if (pe_)
        delete [] pe_;
}

void PositionEmbeding::forward(ForwardData* in, ForwardData* out)
{
    for (int y = 0; y < in->height_; y++) {
        for (int i = 0; i < in->width_; i += 2) {
            out->matrix_[y * in->width_ + i] = in->matrix_[y * in->width_ + i] + pe_[y * word_dim_ + i];
            out->matrix_[y * in->width_ + i + 1] = in->matrix_[y * in->width_ + i + 1] + pe_[y * word_dim_ + i + 1];
        }
    }

    return;
}

void PositionEmbeding::calcPositionEmbeding()
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
