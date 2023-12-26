#ifndef __TRANSFORMER__
#define __TRANSFORMER__

#include "haipin_define.h"
#include "encoder.h"
#include "decoder.h"
#include "ffn_layer.h"
#include "softmax.h"

class Transformer
{
public:
    Transformer(int batch_dim, int word_dim,
                int in_sentence_dim, int gu_sentence_dim,
                int in_vocab_size, int gu_vocab_size);
    ~Transformer();
    void forward(ForwardData* in_fd, ForwardData* gu_fd, ForwardData* out_fd);

private:
    void createFrameWork(int batch_dim, int word_dim, int in_sentence_dim, int gu_sentence_dim, int in_vocab_size, int gu_vocab_size);

private:
    Encoder* encoder_[6];
    Decoder* decoder_[6];

    FfnLayer* ffn_layer_;
    SoftMax* sofx_max_;

    float* tmp_en_matrix_;
    float* tmp_de_matrix_;
};

#endif
