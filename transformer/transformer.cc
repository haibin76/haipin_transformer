#include <math.h>
#include <vector>
#include "transformer.h"

using namespace std;
Transformer::Transformer(int batch_dim, int word_dim, int in_sentence_dim, int gu_sentence_dim, int in_vocab_size, int gu_vocab_size)
{
    if (word_dim != 512)
        printf("WARNNING: the word dimension:%d is not 512!\n", word_dim);
    if ((word_dim <= 0) || (word_dim & 31)) {
        printf("WARNNING: the word dimension:%d is not the 32 multiple\n", word_dim);
    }

    createFrameWork(batch_dim, word_dim, in_sentence_dim, gu_sentence_dim, in_vocab_size, gu_vocab_size);

    return;
}

Transformer::~Transformer()
{
    if (sofx_max_)
        delete sofx_max_;
    if (ffn_layer_)
        delete ffn_layer_;

    for (int y = 0; y < 6; y++) {
        delete[] encoder_[y];
        delete[] decoder_[y];
    }

    return;
}

void Transformer::forward(ForwardData* in_fd, ForwardData* gu_fd, ForwardData* out_fd)
{
    ForwardData tmp_in = *in_fd, tmp_out = *gu_fd, kv;
    tmp_out.matrix_ = tmp_en_matrix_;
    for (int i = 0; i < 6; i++) {
        encoder_[i]->forward(in_fd, in_fd);
    }
    kv = *in_fd;

    for (int i = 0; i < 6; i++) {
        decoder_[i]->forward(gu_fd, &kv, gu_fd);
    }

    ffn_layer_->forward(gu_fd, out_fd);
    sofx_max_->forward(out_fd, out_fd);

    return;
}

void Transformer::createFrameWork(int batch_dim, int word_dim, int in_sentence_dim, int gu_sentence_dim, int in_vocab_size, int gu_vocab_size)
{
    for (int y = 0; y < 6; y++) {
        encoder_[y] = new Encoder(batch_dim, in_sentence_dim, word_dim);
        decoder_[y] = new Decoder(batch_dim, gu_sentence_dim, word_dim);
    }

    ffn_layer_ = new FfnLayer(word_dim, gu_vocab_size);
    sofx_max_ = new SoftMax(word_dim);

    return;
}
