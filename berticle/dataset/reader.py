import numpy as np
import spacy

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import (ArrayField, Field, TextField,
                                  MetadataField, SequenceLabelField)
from allennlp.data.tokenizers import Token
from collections import namedtuple
from typing import Iterator, List

from .bert import BERT_MAX_LENGTH, token_indexer, tokenizer


Tokens = List[spacy.tokens.Token]
ExtendedToken = namedtuple('ExtendedToken',
                           ['label', 'text', 'tag', 'position'])


class ArticleDatasetReader(DatasetReader):
    def __init__(self, token_indexers, tokenizer) -> None:
        super().__init__(lazy=False)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers

    def extend(self, tokens: Tokens) -> Iterator[ExtendedToken]:
        label = 0
        for token in tokens:
            if token.text == "the":
                label = 2
            elif token.text in ("a", "an"):
                label = 1
            else:
                morphems = self._tokenizer(token.text.lower())
                for morphem in morphems:
                    # all the morphems of the token will have the same position
                    # as the original token (we actually don't care about anything)
                    # beyond the very first one
                    yield ExtendedToken(label, morphem, token.pos_, token.idx)
                    label = 0

    @staticmethod
    def generate_mask(tokens: List[ExtendedToken]) -> np.ndarray:
        # original mask that includes only the words that do have articles
        mask = np.array([token.label != 0 for token in tokens])
        # only these tags can have (or not have) articles
        # all tokens with these tag we call "interesting"
        interesting_tags = {'ADJ', 'NOUN', 'PROPN'}
        # all interesting tokens
        interesting_mask = np.array([token.tag in interesting_tags
                                     for token in tokens])
        # it is important for us to include other words as well, so our model
        # knows that other words shouldn't have articles
        other_mask = ~interesting_mask
        # remove words that already have labels
        interesting_mask &= ~mask

        indices = np.arange(len(mask))
        # we want to add 'size' number of interesting words
        # and 'size' number of other words to the mask
        size = np.sum(mask) // 2

        def extend_mask(extension_mask):
            # can't randomly choose more samples than we have in total
            actual_size = min(size, np.sum(extension_mask))
            mask[np.random.choice(indices[extension_mask],
                                  replace=False,
                                  size=actual_size)] = True

        extend_mask(interesting_mask)
        extend_mask(other_mask)

        return mask

    def text_to_instance(self,
                         initial_tokens: List[spacy.tokens.Token]) -> Instance:
        fields = {}  # type: Dict[str, Field]
        extended_tokens = list(self.extend(initial_tokens))
        # There is no point in trying to save more than this amount of tokens.
        # 2 tokens are reserved for special BERT tokens.
        extended_tokens = extended_tokens[:BERT_MAX_LENGTH - 2]

        text_field = TextField([Token(token.text)
                                for token in extended_tokens],
                               self._token_indexers)
        fields['tokens'] = text_field

        labels = [token.label for token in extended_tokens]
        fields['labels'] = SequenceLabelField(labels,
                                              sequence_field=text_field)

        metadata = [token.position for token in extended_tokens]
        fields['positions'] = MetadataField(metadata=metadata)

        mask = self.generate_mask(extended_tokens)
        fields['mask'] = ArrayField(array=mask)

        return Instance(fields)

    def _read(self, filepath: str) -> Iterator[Instance]:
        with open(filepath) as f:
            nlp = spacy.load("en_core_web_sm")
            # we only need a tokenizer and a POS tagger
            for doc in nlp.pipe(f.readlines(),
                                disable=["parser", "ner", "textcat"]):
                yield self.text_to_instance(doc)


def get_default_reader() -> ArticleDatasetReader:
    return ArticleDatasetReader({"tokens": token_indexer}, tokenizer)
