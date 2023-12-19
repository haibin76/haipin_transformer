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
    Transformer(int en_max_word_in_sentence, int word_dim, int de_max_word_in_sentence,
                int en_voc_size, int de_voc_size);
    ~Transformer();
    void forward(ForwardData* en_in, ForwardData* de_in, ForwardData* out);

private:
    void createFrameWork(int en_max_word, int word_dim, int de_max_word, int en_voc_size, int de_voc_size);

private:
    int en_max_word_;
    int de_max_word_;
    int word_dim_;

    Encoder* encoder_[6];
    Decoder* decoder_[6];

    FfnLayer* ffn_layer_;
    SoftMax* sofx_max_;

    float* tmp_en_matrix_;
    float* tmp_de_matrix_;
};

#endif
