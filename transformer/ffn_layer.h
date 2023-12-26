#ifndef __FFN_LAYER__
#define __FFN_LAYER__

#include "haipin_define.h"

class FfnLayer
{
public:
    FfnLayer(int in_dim, int out_dim);
    ~FfnLayer();
    bool forward(ForwardData* in_fd, ForwardData* out_fd);

private:
    void createFrameWork(int in_dim, int out_dim);
    void deleteTempFD();
    void createTempFD(int batch_num, int height, int width);
private:
    //权值矩阵一般没有batch num的概念，只有宽高参数，如果一个batch num
    //乘以该权值矩阵，那么就会给像张量一样[batch num][height][width] * [wm_height_][wm_width_]
    //= [batch num][height][wm_width_],其中width = wm_height_
    int wm_height_;
    int wm_width_;
    float* weight_matrix_;

    //调用该类进行forward，有可能是in_fd == out_fd,如果该类不做判断，有可能再计算的过程中
    //in_fd的值给修改了，而计算的out_fd的值也不对，所以要增加一些函数做判断
    ForwardData  temp_fd_;
};

#endif
