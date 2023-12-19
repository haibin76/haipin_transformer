#ifndef __ADD_NORM_H__
#define __ADD_NORM_H__

#include "haipin_define.h"

class AddLayerNorm
{
public:
    AddLayerNorm(int dim);
    ~AddLayerNorm();
    void forward(ForwardData* in, ForwardData* out);

private:
    int dim_;

    //don't know why gamme beta is 1, 0, but dont update
    char* gamme_;
    char* beta_;
};

#endif
