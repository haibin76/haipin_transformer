#ifndef __ATTENTION_H__
#define __ATTENTION_H__

#include "haipin_define.h"
#include "ffn_layer.h"

class Attention
{
public:
    //batch_dim is the max batch num
    Attention(int batch_dim, int sentence_dim, int word_dim);
    ~Attention();
    void forward(ForwardData* q_fd, ForwardData* k_fd, ForwardData* v_fd, ForwardData* mask_fd, ForwardData* out_fd);

private:
    void createFrameWork(int batch_dim, int sentence_dim, int word_dim);

private:
    int             sentence_dim_;
    ForwardData     fd_;
};

#endif
