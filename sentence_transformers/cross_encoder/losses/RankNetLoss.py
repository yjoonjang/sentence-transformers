from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.util import fullname


class RankNetLoss(nn.Module):
    def __init__(
        self,
        model: CrossEncoder,
        sigma: float = 1.0,
        eps: float = 1e-10,
        activation_fct: nn.Module | None = nn.Identity(),
        mini_batch_size: int | None = None,
    ) -> None:
        """
        RankNet loss implementation for learning to rank. This loss function implements the RankNet algorithm,
        which learns a ranking function by optimizing pairwise document comparisons using a neural network.
        The implementation is optimized to handle padded documents efficiently by only processing valid
        documents during model inference.

        Args:
            model (CrossEncoder): CrossEncoder model to be trained
            sigma (float): Score difference weight used in sigmoid (default: 1.0)
            eps (float): Small constant for numerical stability (default: 1e-10)
            activation_fct (:class:`~torch.nn.Module`): Activation function applied to the logits before computing the
                loss. Defaults to :class:`~torch.nn.Identity`.
            mini_batch_size (int, optional): Number of samples to process in each forward pass. This has a significant
                impact on the memory consumption and speed of the training process. Three cases are possible:
                - If ``mini_batch_size`` is None, the ``mini_batch_size`` is set to the batch size.
                - If ``mini_batch_size`` is greater than 0, the batch is split into mini-batches of size ``mini_batch_size``.
                - If ``mini_batch_size`` is <= 0, the entire batch is processed at once.
                Defaults to None.

        References:
            - Learning to Rank using Gradient Descent: https://icml.cc/Conferences/2015/wp-content/uploads/2015/06/icml_ranking.pdf

        Requirements:
            1. Query with multiple documents (pairwise approach)
            2. Documents must have relevance scores/labels. Both binary and continuous labels are supported.

        Inputs:
            +----------------------------------------+--------------------------------+-------------------------------+
            | Texts                                  | Labels                         | Number of Model Output Labels |
            +========================================+================================+===============================+
            | (query, [doc1, doc2, ..., docN])       | [score1, score2, ..., scoreN]  | 1                             |
            +----------------------------------------+--------------------------------+-------------------------------+

        Example:
            ::

                from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, losses
                from datasets import Dataset

                model = CrossEncoder("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "query": ["What are pandas?", "What is the capital of France?"],
                    "docs": [
                        ["Pandas are a kind of bear.", "Pandas are kind of like fish."],
                        ["The capital of France is Paris.", "Paris is the capital of France.", "Paris is quite large."],
                    ],
                    "labels": [[1, 0], [1, 1, 0]],
                })
                loss = losses.RankNetLoss(model)

                trainer = CrossEncoderTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.sigma = sigma
        self.eps = eps
        self.activation_fct = activation_fct
        self.mini_batch_size = mini_batch_size

        if self.model.num_labels != 1:
            raise ValueError(
                f"{self.__class__.__name__} supports a model with 1 output label, "
                f"but got a model with {self.model.num_labels} output labels."
            )

    def forward(self, inputs: list[list[str], list[list[str]]], labels: list[Tensor]) -> Tensor:
        """
        Compute RankNet loss for a batch of queries and their documents.

        Args:
            inputs: List of (queries, documents_list)
            labels: Ground truth relevance scores, shape (batch_size, num_documents)

        Returns:
            Tensor: RankNet loss over the batch
        """
        if isinstance(labels, Tensor):
            raise ValueError(
                "RankNetLoss expects a list of labels for each sample, but got a single value for each sample."
            )
        if len(inputs) != 2:
            raise ValueError(f"RankNetLoss expects two inputs (queries, documents_list), but got {len(inputs)} inputs.")

        queries, docs_list = inputs
        docs_per_query = [len(docs) for docs in docs_list]
        max_docs = max(docs_per_query)
        batch_size = len(queries)

        if docs_per_query != [len(labels) for labels in labels]:
            raise ValueError(
                f"Number of documents per query in inputs ({docs_per_query}) does not match number of labels per query ({[len(labels) for labels in labels]})."
            )

        # Create input pairs for the model₩
        pairs = [(query, document) for query, docs in zip(queries, docs_list) for document in docs]

        if not pairs:
            # Handle edge case where all documents are padded
            return torch.tensor(0.0, device=self.model.device, requires_grad=True)

        mini_batch_size = self.mini_batch_size or batch_size
        if mini_batch_size <= 0:
            mini_batch_size = len(pairs)

        logits_list = []
        for i in range(0, len(pairs), mini_batch_size):
            mini_batch_pairs = pairs[i : i + mini_batch_size]

            tokens = self.model.tokenizer(
                mini_batch_pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            tokens = tokens.to(self.model.device)

            logits = self.model(**tokens)[0].view(-1)
            logits_list.append(logits)

        logits = torch.cat(logits_list, dim=0)
        logits = self.activation_fct(logits)

        # Create output tensor filled with 0 (padded logits will be ignored via labels)
        logits_matrix = torch.full((batch_size, max_docs), -1e16, device=self.model.device)

        # Place logits in the desired positions in the logit matrix
        doc_indices = torch.cat([torch.arange(len(docs)) for docs in docs_list], dim=0)
        batch_indices = torch.repeat_interleave(torch.arange(batch_size), torch.tensor(docs_per_query))
        logits_matrix[batch_indices, doc_indices] = logits

        # Create labels matrix
        labels_matrix = torch.full_like(logits_matrix, float("-inf"))
        labels_matrix[batch_indices, doc_indices] = torch.cat(labels, dim=0).float()
        labels_matrix = labels_matrix.to(self.model.device)

        # Calculate pairwise differences for scores and labels
        score_diffs = logits_matrix[:, :, None] - logits_matrix[:, None, :]
        label_diffs = labels_matrix[:, :, None] - labels_matrix[:, None, :]

        # Create mask for valid pairs (where both documents are not padded)
        valid_pairs = torch.isfinite(label_diffs)
        
        # Create mask for pairs where l_i > l_j
        positive_pairs = label_diffs > 0

        # Calculate probabilities and target probabilities
        P_ij = torch.sigmoid(self.sigma * score_diffs)
        P_ij = torch.clamp(P_ij, min=self.eps, max=1-self.eps)
        
        # Calculate loss only for pairs where l_i > l_j (positive_pairs)
        # This follows the TensorFlow Ranking implementation more closely
        losses = -torch.log(P_ij)

        # Apply masks and compute mean loss
        masked_loss = losses[valid_pairs & positive_pairs]

        # Handle case when there are no positive pairs
        if masked_loss.numel() == 0:
            return torch.tensor(0.0, device=self.model.device, requires_grad=True)

        loss = torch.mean(masked_loss)

        return loss

    def get_config_dict(self) -> dict[str, float | int | str | None]:
        """
        Get configuration parameters for this loss function.

        Returns:
            Dictionary containing the configuration parameters
        """
        return {
            "sigma": self.sigma,
            "eps": self.eps,
            "activation_fct": fullname(self.activation_fct),
            "mini_batch_size": self.mini_batch_size,
        }

    @property
    def citation(self) -> str:
        return """
@inproceedings{burges2005learning,
  title={Learning to rank using gradient descent},
  author={Burges, Chris and Shaked, Tal and Renshaw, Erin and Lazier, Ari and Deeds, Matt and Hamilton, Nicole and Hullender, Greg},
  booktitle={Proceedings of the 22nd international conference on Machine learning},
  pages={89--96},
  year={2005}
}
""" 