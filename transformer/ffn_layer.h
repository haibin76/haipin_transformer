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
    //Ȩֵ����һ��û��batch num�ĸ��ֻ�п�߲��������һ��batch num
    //���Ը�Ȩֵ������ô�ͻ��������һ��[batch num][height][width] * [wm_height_][wm_width_]
    //= [batch num][height][wm_width_],����width = wm_height_
    int wm_height_;
    int wm_width_;
    float* weight_matrix_;

    //���ø������forward���п�����in_fd == out_fd,������಻���жϣ��п����ټ���Ĺ�����
    //in_fd��ֵ���޸��ˣ��������out_fd��ֵҲ���ԣ�����Ҫ����һЩ�������ж�
    ForwardData  temp_fd_;
};

#endif
