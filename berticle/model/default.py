import torch

from .model import ArticleModel


def get_default_model(load: bool=True,
                      use_gpu: bool=torch.cuda.is_available()) -> ArticleModel:
    model = ArticleModel(
        word_embeddings,
        encoder,
        20
    )

    if load:
        with open("model.th", 'rb') as f:
            model.load_state_dict(torch.load(f))

    if use_gpu:
        model.cuda()

    return model
