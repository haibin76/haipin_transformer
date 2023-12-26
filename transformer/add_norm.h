#ifndef __ADD_NORM_H__
#define __ADD_NORM_H__

#include "haipin_define.h"

class AddLayerNorm
{
public:
    AddLayerNorm(int word_dim);
    ~AddLayerNorm();
    void forward(ForwardData* in_fd, ForwardData* out_fd);

private:

    //don't know why gamme beta is 1, 0, but dont update
    int     word_dim_;
    char*   gamme_;
    char*   beta_;
};

#endif
