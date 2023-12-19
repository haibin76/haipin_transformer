#ifndef __SCALED_ATTENTION_H__
#define __SCALED_ATTENTION_H__

#include <vector>

class ScaledDotProductAttention
{
public:
    ScaledDotProductAttention(int dim);
    ~ScaledDotProductAttention();
    void forward(std::vector<float*>& sentence);

private:
    int dim_;
};

#endif
