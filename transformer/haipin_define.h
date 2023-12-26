#ifndef __HAIPIN_DEFINE_H__
#define __HAIPIN_DEFINE_H__

typedef struct tagForwardData
{
    //the matrix size = height_ * width_ * batch_size_ * sizeof(float)
    float*  matrix_;
    int     height_;
    int     num_;
    int     width_;
    int     batch_num_;
}ForwardData;

#endif
