#ifndef __POSITION_EMBEDING__
#define __POSITION_EMBEDING__

#include <vector>

class PositionEmbeding
{
public:
    PositionEmbeding(int max_word, int word_dim);
    ~PositionEmbeding();
    void forward(std::vector<float*>& sentence);

private:
    void calcPositionEmbeding();

private:
    int max_word_;
    int word_dim_;
    float* pe_;
};

#endif
