# text to vectors

import numpy as np


class Embedder:
    def __init__(self, model_name: str):
        # import in here so I can import the rest of the code (and run the tests)
        # without loading the heavy model
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # normalised so dot product = cosine
            show_progress_bar=False,
        )
        return vectors.astype("float32")
