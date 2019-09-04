import numpy as np

from allennlp.predictors import SentenceTaggerPredictor
from scipy.special import softmax

from berticle.model import get_default_model
from berticle.dataset import get_default_reader


class Predictor:
    def __init__(self) -> None:
        self.model = get_default_model()
        self.model.eval()
        self.predictor = SentenceTaggerPredictor(
            self.model, dataset_reader=get_default_reader()
        )

    def predict(self, sentence):
        logits = np.array(self.predictor.predict(sentence)['class_logits'])
        print(softmax(logits, axis=1))
        articles = np.argmax(softmax(logits, axis=1), axis=1)
        result = np.array(["<null>"] * len(articles))
        result[articles == 1] = "a"
        result[articles == 2] = "the"
        print(result)
