#include <math.h>
#include <vector>
#include "transformer.h"

using namespace std;
Transformer::Transformer(int en_max_word_in_sentence, int word_dim, int de_max_word_in_sentence,
                         int en_voc_size, int de_voc_size)
{
    if (word_dim != 512)
        printf("WARNNING: the word dimension:%d is not 512!\n", word_dim);
    if ((word_dim <= 0) || (word_dim & 31)) {
        printf("WARNNING: the word dimension:%d is not the 32 multiple\n", word_dim);
    }

    en_max_word_ = en_max_word_in_sentence;
    word_dim_ = word_dim;
    de_max_word_ = de_max_word_in_sentence;

    createFrameWork(en_max_word_in_sentence, word_dim, de_max_word_in_sentence, en_voc_size, de_voc_size);

    return;
}

Transformer::~Transformer()
{
    if (sofx_max_)
        delete sofx_max_;
    if (ffn_layer_)
        delete ffn_layer_;
    if (tmp_de_matrix_)
        delete[] tmp_de_matrix_;
    if (tmp_en_matrix_)
        delete[] tmp_en_matrix_;

    for (int y = 0; y < 6; y++) {
        delete[] encoder_[y];
        delete[] decoder_[y];
    }

    return;
}

void Transformer::forward(ForwardData* en_in, ForwardData* de_in, ForwardData* out)
{
    ForwardData tmp_in = *en_in, tmp_out = *en_in, kv;
    tmp_out.matrix_ = tmp_en_matrix_;
    for (int i = 0; i < 6; i++) {
        encoder_[i]->forward(&tmp_in, &tmp_out);
        ForwardData fd = tmp_out;
        tmp_out = tmp_in;
        tmp_in = fd;
    }
    kv = tmp_out;

    tmp_in = *de_in;
    tmp_out = *de_in;
    tmp_out.matrix_ = tmp_de_matrix_;
    for (int i = 0; i < 6; i++) {
        decoder_[i]->forward(&tmp_in, &kv, &tmp_out);
        ForwardData fd = tmp_out;
        tmp_out = tmp_in;
        tmp_in = fd;
    }

    ffn_layer_->forward(&tmp_out, out);
    sofx_max_->forward(out, out);

    return;
}

void Transformer::createFrameWork(int en_max_word, int word_dim, int de_max_word, int en_voc_size, int de_voc_size)
{
    for (int y = 0; y < 6; y++) {
        encoder_[y] = new Encoder(en_max_word, word_dim);
        decoder_[y] = new Decoder(de_max_word, word_dim);
    }

    ffn_layer_ = new FfnLayer(word_dim, de_voc_size);

    sofx_max_ = new SoftMax(word_dim);

    tmp_en_matrix_ = new float[en_max_word * word_dim];
    tmp_de_matrix_ = new float[de_max_word * word_dim];

    return;
}
