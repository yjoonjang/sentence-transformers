from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.util import fullname


class BaseWeighingScheme(nn.Module):
    """Base class for implementing weighing schemes in LambdaLoss."""

    def forward(self, gain: Tensor, discount: Tensor, true_sorted: Tensor) -> Tensor:
        """
        Calculate weights for the loss function.

        Args:
            gain: Normalized gains tensor
            discount: Discount tensor
            true_sorted: Sorted ground truth labels

        Returns:
            Tensor: Calculated weights for the loss
        """
        raise NotImplementedError


class NoWeighingScheme(BaseWeighingScheme):
    """Implementation of no weighing scheme (weights = 1.0)."""

    def forward(self, gain: Tensor, discount: Tensor, true_sorted: Tensor) -> Tensor:
        return torch.tensor(1.0, device=gain.device)


class NDCGLoss1Scheme(BaseWeighingScheme):
    """Implementation of NDCG Loss1 weighing scheme."""

    def forward(self, gain: Tensor, discount: Tensor, true_sorted: Tensor) -> Tensor:
        return (gain / discount)[:, :, None]


class NDCGLoss2Scheme(BaseWeighingScheme):
    """Implementation of NDCG Loss2 weighing scheme."""

    def forward(self, gain: Tensor, discount: Tensor, true_sorted: Tensor) -> Tensor:
        pos_idxs = torch.arange(1, gain.shape[1] + 1, device=gain.device)
        delta_idxs = torch.abs(pos_idxs[:, None] - pos_idxs[None, :])
        deltas = torch.abs(
            torch.pow(torch.abs(discount[0, delta_idxs - 1]), -1.0)
            - torch.pow(torch.abs(discount[0, delta_idxs]), -1.0)
        )
        deltas.diagonal().zero_()
        return deltas[None, :, :] * torch.abs(gain[:, :, None] - gain[:, None, :])


class LambdaRankScheme(BaseWeighingScheme):
    """Implementation of LambdaRank weighing scheme."""

    def forward(self, gain: Tensor, discount: Tensor, true_sorted: Tensor) -> Tensor:
        return torch.abs(torch.pow(discount[:, :, None], -1.0) - torch.pow(discount[:, None, :], -1.0)) * torch.abs(
            gain[:, :, None] - gain[:, None, :]
        )


class NDCGLoss2PPScheme(BaseWeighingScheme):
    """Implementation of NDCG Loss2++ weighing scheme."""

    def __init__(self, mu: float = 10.0):
        super().__init__()
        self.mu = mu
        self.ndcg_loss2 = NDCGLoss2Scheme()
        self.lambda_rank = LambdaRankScheme()

    def forward(self, gain: Tensor, discount: Tensor, true_sorted: Tensor) -> Tensor:
        ndcg_weights = self.ndcg_loss2(gain, discount, true_sorted)
        lambda_weights = self.lambda_rank(gain, discount, true_sorted)
        return self.mu * ndcg_weights + lambda_weights


class LambdaLoss(nn.Module):
    def __init__(
        self,
        model: CrossEncoder,
        weighing_scheme: BaseWeighingScheme | None = NoWeighingScheme(),
        k: int | None = None,
        sigma: float = 1.0,
        eps: float = 1e-10,
        reduction_log: Literal["natural", "binary"] = "binary",
        activation_fct: nn.Module | None = nn.Identity(),
        mini_batch_size: int | None = None,
    ) -> None:
        """
        The LambdaLoss Framework for Ranking Metric Optimization. This loss function implements the LambdaLoss framework for ranking metric optimization,
        which provides various weighing schemes including LambdaRank and NDCG variations.
        The implementation is optimized to handle padded documents efficiently by only
        processing valid documents during model inference.

        .. note::

            The number of documents per query can vary between samples with the ``LambdaLoss``.

        Args:
            model (CrossEncoder): CrossEncoder model to be trained
            weighing_scheme (:class: `BaseWeighingScheme`, optional): Weighing scheme to use for the loss.
                - NoWeighingScheme: No weighing scheme (weights = 1.0)
                - NDCGLoss1Scheme: NDCG Loss1 weighing scheme
                - NDCGLoss2Scheme: NDCG Loss2 weighing scheme
                - LambdaRankScheme: LambdaRank weighing scheme
                - NDCGLoss2PPScheme: NDCG Loss2++ weighing scheme

                Defaults to NoWeighingScheme.
            k (int, optional): Number of documents to consider for NDCG@K. Defaults to None (use all documents).
            sigma (float): Score difference weight used in sigmoid
            eps (float): Small constant for numerical stability
            reduction_log (str): Type of logarithm to use
                - "natural": Natural logarithm (log)
                - "binary": Binary logarithm (log2)
            activation_fct (:class:`~torch.nn.Module`): Activation function applied to the logits before computing the
                loss. Defaults to :class:`~torch.nn.Identity`.
            mini_batch_size (int, optional): Number of samples to process in each forward pass. This has a significant
                impact on the memory consumption and speed of the training process. Three cases are possible:

                - If ``mini_batch_size`` is None, the ``mini_batch_size`` is set to the batch size.
                - If ``mini_batch_size`` is greater than 0, the batch is split into mini-batches of size ``mini_batch_size``.
                - If ``mini_batch_size`` is <= 0, the entire batch is processed at once.

                Defaults to None.

        References:
            - The LambdaLoss Framework for Ranking Metric Optimization: https://marc.najork.org/papers/cikm2018.pdf
            - `Training Examples > Learning to Rank <../../../examples/cross-encoder/training/ms_marco/training_ms_marco_lambda.py>`_

        Requirements:
            1. Query with multiple documents (listwise approach)
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
                loss = losses.LambdaLoss(model)

                trainer = CrossEncoderTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.weighing_scheme = weighing_scheme or NoWeighingScheme()
        self.k = k
        self.sigma = sigma
        self.eps = eps
        self.reduction_log = reduction_log
        self.activation_fct = activation_fct or nn.Identity()
        self.mini_batch_size = mini_batch_size

        if self.model.num_labels != 1:
            raise ValueError(
                f"{self.__class__.__name__} supports a model with 1 output label, "
                f"but got a model with {self.model.num_labels} output labels."
            )

    def forward(self, inputs: list[list[str], list[list[str]]], labels: list[Tensor]) -> Tensor:
        """
        Compute LambdaLoss for a batch of queries and their documents.

        Args:
            inputs: List of (queries, documents_list)
            labels: Ground truth relevance scores, shape (batch_size, num_documents)

        Returns:
            Tensor: LambdaLoss loss over the batch
        """
        if isinstance(labels, Tensor):
            raise ValueError(
                "LambdaLoss expects a list of labels for each sample, but got a single value for each sample."
            )
        if len(inputs) != 2:
            raise ValueError(f"LambdaLoss expects two inputs (queries, documents_list), but got {len(inputs)} inputs.")

        queries, docs_list = inputs
        docs_per_query = [len(docs) for docs in docs_list]
        max_docs = max(docs_per_query)
        batch_size = len(queries)

        if docs_per_query != [len(labels) for labels in labels]:
            raise ValueError(
                f"Number of documents per query in inputs ({docs_per_query}) does not match number of labels per query ({[len(labels) for labels in labels]})."
            )

        # Create input pairs for the model
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

        # Idem for labels, but fill with -inf to 0 out padded logits in the loss
        labels_matrix = torch.full_like(logits_matrix, float("-inf"))
        labels_matrix[batch_indices, doc_indices] = torch.cat(labels, dim=0).float()
        labels_matrix = labels_matrix.to(self.model.device)

        # Calculate LambdaLoss components
        logits_matrix_sorted, indices_pred = logits_matrix.sort(descending=True, dim=-1)
        labels_matrix_sorted, _ = labels_matrix.sort(descending=True, dim=-1)

        # Create masks for valid pairs
        true_sorted_by_preds = torch.gather(labels_matrix, dim=1, index=indices_pred)
        true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
        padded_pairs_mask = torch.isfinite(true_diffs)

        if not isinstance(self.weighing_scheme, NDCGLoss1Scheme):
            padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)

        # Create truncation mask if k is specified
        k = self.k or max_docs
        ndcg_at_k_mask = torch.zeros((max_docs, max_docs), dtype=torch.bool, device=self.model.device)
        ndcg_at_k_mask[:k, :k] = 1

        # Calculate gains and discounts
        true_sorted_by_preds.clamp_(min=0.0)
        labels_matrix_sorted.clamp_(min=0.0)

        pos_idxs = torch.arange(1, max_docs + 1).to(self.model.device)
        discount = torch.log2(1.0 + pos_idxs.float())[None, :]
        maxDCGs = torch.sum(((torch.pow(2, labels_matrix_sorted) - 1) / discount)[:, :k], dim=-1).clamp(min=self.eps)
        gain = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

        # Apply weighing scheme
        weights = self.weighing_scheme(gain, discount, true_sorted_by_preds)

        # Calculate scores differences and probabilities
        scores_diffs = (logits_matrix_sorted[:, :, None] - logits_matrix_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
        scores_diffs.masked_fill_(torch.isnan(scores_diffs), 0.0)
        weighted_probas = (torch.sigmoid(self.sigma * scores_diffs).clamp(min=self.eps) ** weights).clamp(min=self.eps)

        # Calculate losses based on specified logarithm base
        if self.reduction_log == "natural":
            losses = torch.log(weighted_probas)
        else:  # binary
            losses = torch.log2(weighted_probas)

        # Apply masks and reduction
        masked_losses = losses[padded_pairs_mask & ndcg_at_k_mask]
        loss = -torch.mean(masked_losses)
        return loss

    def get_config_dict(self) -> dict[str, float | int | str | None]:
        """
        Get configuration parameters for this loss function.

        Returns:
            Dictionary containing the configuration parameters
        """
        return {
            "weighing_scheme": fullname(self.weighing_scheme),
            "k": self.k,
            "sigma": self.sigma,
            "eps": self.eps,
            "reduction_log": self.reduction_log,
            "activation_fct": fullname(self.activation_fct),
            "mini_batch_size": self.mini_batch_size,
        }

    @property
    def citation(self) -> str:
        return """
@inproceedings{wang2018lambdaloss,
  title={The lambdaloss framework for ranking metric optimization},
  author={Wang, Xuanhui and Li, Cheng and Golbandi, Nadav and Bendersky, Michael and Najork, Marc},
  booktitle={Proceedings of the 27th ACM international conference on information and knowledge management},
  pages={1313--1322},
  year={2018}
}
"""
