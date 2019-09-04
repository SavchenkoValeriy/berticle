from allennlp.data.token_indexers import PretrainedBertIndexer

BERT_MAX_LENGTH=512

token_indexer = PretrainedBertIndexer(
    pretrained_model="bert-base-uncased",
    max_pieces=BERT_MAX_LENGTH,
    do_lowercase=True,
 )

def tokenizer(s: str):
    return token_indexer.wordpiece_tokenizer(s)
