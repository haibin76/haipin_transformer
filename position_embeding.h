#ifndef __POSITION_EMBEDING__
#define __POSITION_EMBEDING__

#include "transformer/haipin_define.h"

class PositionEmbeding
{
public:
    PositionEmbeding(int sentence_dim, int word_dim);
    ~PositionEmbeding();
    void forward(ForwardData* in, ForwardData* out);

private:
    void calcPositionEmbeding();

private:
    int     sentence_dim_;
    int     word_dim_;
    float*  pe_;
};

#endif
