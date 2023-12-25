#ifndef __POSITION_EMBEDING__
#define __POSITION_EMBEDING__

#include "transformer/haipin_define.h"

class PositionEmbeding
{
public:
    PositionEmbeding(int max_word, int word_dim);
    ~PositionEmbeding();
    void forward(ForwardData* in, ForwardData* out);

private:
    void calcPositionEmbeding();

private:
    int max_word_;
    int word_dim_;
    float* pe_;
};

#endif
