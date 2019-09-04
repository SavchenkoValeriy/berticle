import numpy as np

from allennlp.predictors import SentenceTaggerPredictor
from scipy.special import softmax

from berticle.model import get_default_model
from berticle.dataset import get_default_reader


def predict(sentence):
    predictor = SentenceTaggerPredictor(get_default_model(),
                                        dataset_reader=get_default_reader())
    logits = np.array(predictor.predict(sentence)['class_logits'])
    articles = np.argmax(softmax(logits, axis=1), axis=1)
    result = np.array(["<null>"] * len(articles))
    result[articles == 1] = "a"
    result[articles == 2] = "the"
    print(result)
