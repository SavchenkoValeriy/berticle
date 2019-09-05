import numpy as np

from allennlp.predictors import Predictor
from scipy.special import softmax

from berticle.model import get_default_model
from berticle.dataset import get_default_reader


class ArticlePredictor(Predictor):
    def __init__(self) -> None:
        model = get_default_model()
        model.eval()
        super().__init__(model, get_default_reader())

    def predict(self, text):
        lines = text.split('\n')
        lines = [line for line in lines if line]
        instances = list(self._dataset_reader.read_lines(lines))
        logits = np.array([result['class_logits']
                           for result in self.predict_batch_instance(instances)])
        print(softmax(logits, axis=2))
        articles = np.argmax(softmax(logits, axis=2), axis=2)
        results = np.full(articles.shape, "<null>")
        results[articles == 1] = "a"
        results[articles == 2] = "the"
        print(results)
        for line, instance, result in zip(lines, instances, results):
            for article, pos in reversed(list(zip(result,
                                                  instance['positions'].metadata))):
                if article in ("a", "the"):
                    line = line[:pos] + article + " " + line[pos:]
            print(line)
