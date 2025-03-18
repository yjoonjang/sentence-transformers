from __future__ import annotations

import torch
from torch import nn
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.losses.PListMLELoss import PListMLELoss


class ListMLELoss(PListMLELoss):
    def __init__(
        self,
        model: CrossEncoder,
        lambda_weight: None = None,
        activation_fct: nn.Module | None = nn.Identity(),
        mini_batch_size: int | None = None,
        respect_input_order: bool = True,
    ) -> None:
        """
        ListMLE loss for learning to rank with position-aware weighting. This loss function implements 
        the ListMLE ranking algorithm which uses a list-wise approach based on maximum likelihood 
        estimation of permutations. It maximizes the likelihood of the permutation induced by the 
        ground truth labels with optional position-aware weighting.

        .. note::

            The number of documents per query can vary between samples with the ``ListMLELoss``.

        Args:
            model (CrossEncoder): CrossEncoder model to be trained
            lambda_weight (ListMLELambdaWeight, optional): Weighting scheme to use. When specified,
                implements Position-Aware ListMLE which applies different weights to different rank 
                positions. Default is None (standard ListMLE).
            activation_fct (:class:`~torch.nn.Module`): Activation function applied to the logits before computing the
                loss. Defaults to :class:`~torch.nn.Identity`.
            mini_batch_size (int, optional): Number of samples to process in each forward pass. This has a significant
                impact on the memory consumption and speed of the training process. Three cases are possible:

                - If ``mini_batch_size`` is None, the ``mini_batch_size`` is set to the batch size.
                - If ``mini_batch_size`` is greater than 0, the batch is split into mini-batches of size ``mini_batch_size``.
                - If ``mini_batch_size`` is <= 0, the entire batch is processed at once.

                Defaults to None.
            respect_input_order (bool): Whether to respect the original input order of documents.
                If True, assumes the input documents are already ordered by relevance (most relevant first).
                If False, sorts documents by label values. Defaults to True.

        References:
            - Listwise approach to learning to rank: theory and algorithm: https://dl.acm.org/doi/abs/10.1145/1390156.1390306
            - `Cross Encoder > Training Examples > MS MARCO <../../../examples/cross_encoder/training/ms_marco/README.html>`_

        Requirements:
            1. Query with multiple documents (listwise approach)
            2. Documents must have relevance scores/labels. Both binary and continuous labels are supported.
            3. Documents must be sorted in a defined rank order.

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
                
                # Standard ListMLE loss respecting input order
                loss = losses.ListMLELoss(model)

                trainer = CrossEncoderTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__(
            model=model,
            lambda_weight=lambda_weight,
            activation_fct=activation_fct,
            mini_batch_size=mini_batch_size,
            respect_input_order=respect_input_order,
        )