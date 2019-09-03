


NUMBER_OF_ARTICLES = 3  # no, a/an, and the

class ArticleModel(Model):
    def __init__(self, word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 hidden_size: int,
                 out_size: int=NUMBER_OF_ARTICLES):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.transformer = MultiHeadSelfAttention(
            num_heads=out_size,
            input_dim=self.encoder.get_output_dim(),
            attention_dim=out_size * 10,
            values_dim=out_size * hidden_size
        )
        self.projection = nn.Linear(self.transformer.get_output_dim(),
                                    out_size)
        self.loss = sequence_cross_entropy_with_logits
        self.accuracy = CategoricalAccuracy()
        self.full_accuracy = CategoricalAccuracy()

    def forward(self, tokens: Dict[str, torch.Tensor],
                mask: torch.Tensor = None,
                labels: torch.Tensor = None,
                positions: List[int] = []) -> torch.Tensor:
        seq_mask = get_text_field_mask(tokens)

        embeddings = self.word_embeddings(tokens)
        state = self.encoder(embeddings, seq_mask)
        state = self.transformer(state, seq_mask)
        class_logits = self.projection(state)

        output = {"class_logits": class_logits}

        if labels is not None and mask is not None:
            self.full_accuracy(class_logits, labels, seq_mask)
            self.accuracy(class_logits, labels, mask)
            output["loss"] = self.loss(class_logits, labels, mask)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset),
                "full_accuracy": self.full_accuracy.get_metric(reset)}
