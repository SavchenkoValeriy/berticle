import torch

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.text_field_embedders import (TextFieldEmbedder,
                                                   BasicTextFieldEmbedder)
from allennlp.modules.token_embedders.bert_token_embedder \
    import PretrainedBertEmbedder
from overrides import overrides


def get_bert_embedder() -> TextFieldEmbedder:
    bert_embedder = PretrainedBertEmbedder(
        pretrained_model="bert-base-uncased",
        top_layer_only=True, # conserve memory
    )
    return BasicTextFieldEmbedder({"tokens": bert_embedder},
                                  # we'll be ignoring masks so we'll need
                                  # to set this to True
                                  allow_unmatched_keys = True)


class BertCLSDropper(Seq2SeqEncoder):
    def forward(self, embs: torch.tensor,
                mask: torch.tensor=None) -> torch.tensor:
        # drop the very first token <CLS> embedding
        return embs[:, 1:-1]
    
    @overrides
    def get_output_dim(self) -> int:
        return get_bert_embedder().get_output_dim()


def get_bert_encoder() -> BertCLSDropper:
    vocab = Vocabulary()
    return BertCLSDropper(vocab)
