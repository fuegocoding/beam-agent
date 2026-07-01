"""TinyStyler 768-dim authorship embedding. Optional — requires torch."""

import logging

from brain_platform.pipeline.brain_file.schema import StyleEmbedding

logger = logging.getLogger(__name__)


class StyleEmbedder:
    def compute(self, texts: list[str]) -> StyleEmbedding:
        if not texts:
            return StyleEmbedding()

        try:
            import torch
            from tinystyler.style_utils import load_style_model, text_to_style

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, tokenizer, _ = load_style_model("AnnaWegmann/Style-Embedding")
            model = model.to(device)

            # Compute embeddings in batches
            all_embeddings = []
            batch_size = 16
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                embeddings = text_to_style(
                    model, tokenizer, batch, device, model_type="style", max_length=512
                )
                all_embeddings.append(embeddings)

            # Mean pool across all samples
            stacked = torch.cat(all_embeddings, dim=0)
            mean_vector = stacked.mean(dim=0)

            return StyleEmbedding(
                authorship_vector=mean_vector.tolist(),
                model="AnnaWegmann/Style-Embedding",
                sample_count=len(texts),
            )

        except ImportError:
            logger.info("torch/tinystyler not available, skipping style embedding")
            return StyleEmbedding(sample_count=len(texts))
        except Exception:
            logger.exception("Style embedding computation failed")
            return StyleEmbedding(sample_count=len(texts))
