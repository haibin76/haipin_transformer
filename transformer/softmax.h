#ifndef __SOFT_MAX_H__
#define __SOFT_MAX_H__

#include "haipin_define.h"

class SoftMax
{
public:
    SoftMax(int dim);
    ~SoftMax();
    void forward(ForwardData* in_fd, ForwardData* out_fd);

private:
    int dim_;

    //don't know why gamme beta is 1, 0, but dont update
};

#endif
