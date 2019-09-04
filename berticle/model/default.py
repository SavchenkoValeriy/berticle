import torch

from .bert import get_bert_embedder, get_bert_encoder
from .model import ArticleModel


def get_default_model(load: bool=True,
                      use_gpu: bool=torch.cuda.is_available()) -> ArticleModel:
    model = ArticleModel(
        get_bert_embedder(),
        get_bert_encoder(),
        20
    )

    if load:
        with open("model.th", 'rb') as f:
            model.load_state_dict(torch.load(f, map_location='cpu'))

    if use_gpu:
        model.cuda()

    return model
