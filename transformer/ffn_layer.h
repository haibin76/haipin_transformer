#ifndef __FFN_LAYER__
#define __FFN_LAYER__

#include "haipin_define.h"

class FfnLayer
{
public:
    FfnLayer(int in_dim, int out_dim);
    ~FfnLayer();
    bool forward(ForwardData* in, ForwardData* out);

private:
    void createMatrix(int in_dim, int out_dim);

private:
    int height_;
    int width_;

    float* matrix_;
};

#endif
