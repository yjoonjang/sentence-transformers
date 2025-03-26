from __future__ import annotations


import logging
import traceback

from datasets import load_dataset

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator


from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments


import torch
from torch import Tensor, nn

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.util import fullname
from sentence_transformers.cross_encoder.data_collator import CrossEncoderDataCollator
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class PrerankerDataCollator(CrossEncoderDataCollator):
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        column_names = list(features[0].keys())

        # We should always be able to return a loss, label or not:
        batch = {}

        if "dataset_name" in column_names:
            column_names.remove("dataset_name")
            batch["dataset_name"] = features[0]["dataset_name"]

        # Extract the label column if it exists
        for label_column in self.valid_label_columns:
            if label_column in column_names:
                # If the label column is a list/tuple/collection, we create a list of tensors
                if isinstance(features[0][label_column], Collection):
                    batch["label"] = [torch.tensor(row[label_column]) for row in features]
                else:
                    # Otherwise, if it's e.g. single values, we create a tensor
                    batch["label"] = torch.tensor([row[label_column] for row in features])
                column_names.remove(label_column)
                break

        # 특별히 bio_labels 처리 추가
        if "bio_labels" in column_names:
            batch["bio_labels"] = [torch.tensor(row["bio_labels"]) for row in features]
            column_names.remove("bio_labels")

        for column_name in column_names:
            # If the prompt length has been set, we should add it to the batch
            if column_name.endswith("_prompt_length") and column_name[: -len("_prompt_length")] in column_names:
                batch[column_name] = torch.tensor([row[column_name] for row in features], dtype=torch.int)
                continue

            batch[column_name] = [row[column_name] for row in features]

        return batch


# CrossEncoderTrainer를 확장하는 커스텀 트레이너 클래스
class PrerankerTrainer(CrossEncoderTrainer):
    def compute_loss(
        self, 
        model, 
        inputs, 
        return_outputs=False,
        num_items_in_batch=None
    ):
        """
        Compute loss 메소드를 오버라이드하여 bio_labels를 손실 함수에 전달합니다.
        """
        dataset_name = inputs.pop("dataset_name", None)

        # bio_labels 추출
        bio_labels = inputs.pop("bio_labels")

        features, labels = self.collect_features(inputs)
        loss_fn = self.loss
        
        if isinstance(loss_fn, dict) and dataset_name:
            loss_fn = loss_fn[dataset_name]

        # Insert the wrapped model into the loss function if needed
        if (
            model == self.model_wrapped
            and model != self.model
            and hasattr(loss_fn, "model")
            and loss_fn.model != model
        ):
            loss_fn.model = model
        
        # loss 함수에 bio_labels도 함께 전달
        loss = loss_fn(features, labels, bio_labels)
        
        if return_outputs:
            return loss, {}
        return loss


class PListMLELambdaWeight(nn.Module):
    """Base class for implementing weighting schemes in Position-Aware ListMLE Loss."""

    def __init__(self, rank_discount_fn=None) -> None:
        """
        Initialize a lambda weight for PListMLE loss.

        Args:
            rank_discount_fn: Function that computes a discount for each rank position.
                              If None, uses default discount of 2^(num_docs - rank) - 1.
        """
        super().__init__()
        self.rank_discount_fn = rank_discount_fn

    def forward(self, mask: Tensor) -> Tensor:
        """
        Calculate position-aware weights for the PListMLE loss.

        Args:
            mask: A boolean mask indicating valid positions [batch_size, num_docs]

        Returns:
            Tensor: Weights for each position [batch_size, num_docs]
        """
        if self.rank_discount_fn is not None:
            return self.rank_discount_fn(mask)

        # Apply default rank discount: 2^(num_docs - rank) - 1
        num_docs_per_query = mask.sum(dim=1, keepdim=True)
        ranks = torch.arange(mask.size(1), device=mask.device).expand_as(mask)
        weights = torch.pow(2.0, num_docs_per_query - ranks) - 1.0
        weights = weights * mask
        return weights


class Preranker_PListMLELoss(nn.Module):
    def __init__(
        self,
        model: CrossEncoder,
        lambda_weight: PListMLELambdaWeight | None = PListMLELambdaWeight(),
        activation_fct: nn.Module | None = nn.Identity(),
        mini_batch_size: int | None = None,
        respect_input_order: bool = True,
    ) -> None:
        """
        PListMLE loss for learning to rank with position-aware weighting. This loss function implements
        the ListMLE ranking algorithm which uses a list-wise approach based on maximum likelihood
        estimation of permutations. It maximizes the likelihood of the permutation induced by the
        ground truth labels with position-aware weighting.

        This loss is also known as Position-Aware ListMLE or p-ListMLE.

        .. note::

            The number of documents per query can vary between samples with the ``PListMLELoss``.

        Args:
            model (CrossEncoder): CrossEncoder model to be trained
            lambda_weight (PListMLELambdaWeight, optional): Weighting scheme to use. When specified,
                implements Position-Aware ListMLE which applies different weights to different rank
                positions. Default is None (standard PListMLE).
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
            - Position-Aware ListMLE: A Sequential Learning Process for Ranking: https://auai.org/uai2014/proceedings/individuals/164.pdf
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

        Recommendations:
            - Use :class:`~sentence_transformers.util.mine_hard_negatives` with ``output_format="labeled-list"``
              to convert question-answer pairs to the required input format with hard negatives.

        Relations:
            - The :class:`~sentence_transformers.cross_encoder.losses.PListMLELoss` is an extension of the
              :class:`~sentence_transformers.cross_encoder.losses.ListMLELoss` and allows for positional weighting
              of the loss. :class:`~sentence_transformers.cross_encoder.losses.PListMLELoss` generally outperforms
              :class:`~sentence_transformers.cross_encoder.losses.ListMLELoss` and is recommended over it.
            - :class:`~sentence_transformers.cross_encoder.losses.LambdaLoss` takes the same inputs, and generally
              outperforms this loss.

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

                # Either: Position-Aware ListMLE with default weighting
                lambda_weight = losses.PListMLELambdaWeight()
                loss = losses.PListMLELoss(model, lambda_weight=lambda_weight)

                # or: Position-Aware ListMLE with custom weighting function
                def custom_discount(ranks): # e.g. ranks: [1, 2, 3, 4, 5]
                    return 1.0 / torch.log1p(ranks)
                lambda_weight = losses.PListMLELambdaWeight(rank_discount_fn=custom_discount)
                loss = losses.PListMLELoss(model, lambda_weight=lambda_weight)

                trainer = CrossEncoderTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.lambda_weight = lambda_weight
        self.activation_fct = activation_fct or nn.Identity()
        self.mini_batch_size = mini_batch_size
        self.respect_input_order = respect_input_order
        self.eps = 1e-10

        if self.model.num_labels != 1:
            raise ValueError(
                f"{self.__class__.__name__} supports a model with 1 output label, "
                f"but got a model with {self.model.num_labels} output labels."
            )

    def forward(self, inputs: list[list[str], list[list[str]]], labels: list[Tensor], bio_labels: list[Tensor] = None) -> Tensor:
        """
        Compute PListMLE loss for a batch of queries and their documents.

        Args:
            inputs: List of (queries, documents_list)
            labels: Ground truth relevance scores, shape (batch_size, num_documents)
            bio_labels: BIO labels for queries, optional

        Returns:
            Tensor: Mean PListMLE loss over the batch
        """
        if isinstance(labels, Tensor):
            raise ValueError(
                "PListMLELoss expects a list of labels for each sample, but got a single value for each sample."
            )

        # if len(inputs) != 2:
            # raise ValueError(
            #     f"PListMLELoss expects two inputs (queries, documents_list), but got {len(inputs)} inputs."
            # )
        print(f"{len(inputs)=}")
        print(f"{len(bio_labels)=}")
        # bio_labels가 전달되었는지 확인하고 출력
        if bio_labels is not None:
            print("Bio labels received in forward method:", len(bio_labels))
        else:
            print("No bio labels received in forward method")
        print(bio_labels)
        queries, docs_list = inputs
        docs_per_query = [len(docs) for docs in docs_list]
        max_docs = max(docs_per_query)
        batch_size = len(queries)

        if docs_per_query != [len(labels) for labels in labels]:
            raise ValueError(
                f"Number of documents per query in inputs ({docs_per_query}) does not match number of labels per query ({[len(labels) for labels in labels]})."
            )

        pairs = [(query, document) for query, docs in zip(queries, docs_list) for document in docs]

        if not pairs:
            # Handle edge case where there are no documents
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

        # Create output tensor filled with a very small value for padded logits
        logits_matrix = torch.full((batch_size, max_docs), 1e-16, device=self.model.device)

        # Place logits in the desired positions in the logit matrix
        doc_indices = torch.cat([torch.arange(len(docs)) for docs in docs_list], dim=0)
        batch_indices = torch.repeat_interleave(torch.arange(batch_size), torch.tensor(docs_per_query))
        logits_matrix[batch_indices, doc_indices] = logits

        # Create a mask for valid entries
        mask = torch.zeros_like(logits_matrix, dtype=torch.bool)
        mask[batch_indices, doc_indices] = True

        # Convert labels to tensor matrix
        labels_matrix = torch.full_like(logits_matrix, -float("inf"))
        labels_matrix[batch_indices, doc_indices] = torch.cat(labels, dim=0).float()

        if not torch.any(mask):
            return torch.tensor(0.0, device=self.model.device, requires_grad=True)

        if not self.respect_input_order:
            # Sort by labels in descending order if not respecting input order.
            sorted_labels, indices = labels_matrix.sort(descending=True, dim=1)
            sorted_logits = torch.gather(logits_matrix, 1, indices)
        else:
            # Use the original input order, assuming it's already ordered by relevance
            sorted_logits = logits_matrix

        # Compute log-likelihood using Plackett-Luce model
        scores = sorted_logits.exp()
        cumsum_scores = torch.flip(torch.cumsum(torch.flip(scores, [1]), 1), [1])
        log_probs = sorted_logits - torch.log(cumsum_scores + self.eps)

        # Apply position-aware lambda weights if specified. If None, then this loss
        # is just ListMLE.
        if self.lambda_weight is not None:
            lambda_weight = self.lambda_weight(mask)
            # Normalize weights to sum to 1
            lambda_weight = lambda_weight / (lambda_weight.sum(dim=1, keepdim=True) + self.eps)
            log_probs = log_probs * lambda_weight

        # Sum the log probabilities for each list and mask padded entries
        log_probs[~mask] = 0.0
        per_query_losses = -torch.sum(log_probs, dim=1)

        if not torch.any(per_query_losses):
            return torch.tensor(0.0, device=self.model.device, requires_grad=True)

        # 여기에서 bio_labels를 활용한 추가 손실을 계산할 수 있습니다
        # 예시: bio_labels가 있을 때 추가 손실을 계산하여 결합
        if bio_labels is not None:
            # 실제 구현에서는 여기에 bio_labels를 활용한 손실 계산 로직을 추가할 수 있습니다
            pass

        # Average loss over all lists
        return torch.mean(per_query_losses)

    def get_config_dict(self) -> dict[str, float | int | str | None]:
        """
        Get configuration parameters for this loss function.

        Returns:
            Dictionary containing the configuration parameters
        """
        return {
            "lambda_weight": None if self.lambda_weight is None else fullname(self.lambda_weight),
            "activation_fct": fullname(self.activation_fct),
            "mini_batch_size": self.mini_batch_size,
            "respect_input_order": self.respect_input_order,
        }

    @property
    def citation(self) -> str:
        return """
@inproceedings{lan2014position,
  title={Position-Aware ListMLE: A Sequential Learning Process for Ranking.},
  author={Lan, Yanyan and Zhu, Yadong and Guo, Jiafeng and Niu, Shuzi and Cheng, Xueqi},
  booktitle={UAI},
  volume={14},
  pages={449--458},
  year={2014}
}
"""



def main():
    model_name = "microsoft/MiniLM-L12-H384-uncased"

    # Set the log level to INFO to get more information
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    # train_batch_size and eval_batch_size inform the size of the batches, while mini_batch_size is used by the loss
    # to subdivide the batch into smaller parts. This mini_batch_size largely informs the training speed and memory usage.
    # Keep in mind that the loss does not process `train_batch_size` pairs, but `train_batch_size * num_docs` pairs.
    train_batch_size = 16
    eval_batch_size = 16
    mini_batch_size = 16
    num_epochs = 1
    max_docs = None
    respect_input_order = True  # Whether to respect the original order of documents

    # 1. Define our CrossEncoder model
    model = CrossEncoder(model_name, num_labels=1)
    print("Model max length:", model.max_length)
    print("Model num labels:", model.num_labels)

    # 2. Load the MS MARCO dataset: https://huggingface.co/datasets/microsoft/ms_marco
    logging.info("Read train dataset")
    dataset = load_dataset("microsoft/ms_marco", "v1.1", split="train")

    def listwise_mapper(batch, max_docs: int | None = 10):
        processed_queries = []
        processed_docs = []
        processed_labels = []
        processed_bio_labels = []

        for query, passages_info in zip(batch["query"], batch["passages"]):
            # Extract passages and labels
            passages = passages_info["passage_text"]
            labels = passages_info["is_selected"]

            # Pair passages with labels and sort descending by label (positives first)
            paired = sorted(zip(passages, labels), key=lambda x: x[1], reverse=True)

            # Separate back to passages and labels
            sorted_passages, sorted_labels = zip(*paired) if paired else ([], [])

            # Filter queries without any positive labels
            if max(sorted_labels) < 1.0:
                continue

            # Truncate to max_docs
            if max_docs is not None:
                sorted_passages = list(sorted_passages[:max_docs])
                sorted_labels = list(sorted_labels[:max_docs])

            processed_queries.append(query)
            processed_docs.append(sorted_passages)
            processed_labels.append(sorted_labels)
            
            # Generate bio_labels for the query
            query_split_by_ws = query.split()
            import random
            bio_labels = [random.randint(0, 11) for _ in range(len(query_split_by_ws))]
            processed_bio_labels.append(bio_labels)

        return {
            "query": processed_queries,
            "docs": processed_docs,
            "labels": processed_labels,
            "bio_labels": processed_bio_labels
        }

    # Create a dataset with a "query" column with strings, a "docs" column with lists of strings,
    # and a "labels" column with lists of floats
    dataset = dataset.map(
        lambda batch: listwise_mapper(batch=batch, max_docs=max_docs),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Processing listwise samples",
    )

    dataset = dataset.train_test_split(test_size=1_000)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    logging.info(train_dataset)

    # 3. Define our training loss
    loss = Preranker_PListMLELoss(model, mini_batch_size=mini_batch_size, respect_input_order=respect_input_order)

    # 4. Define the evaluator. We use the CENanoBEIREvaluator, which is a light-weight evaluator for English reranking
    evaluator = CrossEncoderNanoBEIREvaluator(dataset_names=["msmarco", "nfcorpus", "nq"], batch_size=eval_batch_size)
    # evaluator(model)

    # 5. Define the training arguments
    short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
    run_name = f"reranker-msmarco-v1.1-{short_model_name}-plistmle"
    args = CrossEncoderTrainingArguments(
        # Required parameter:
        output_dir=f"models/{run_name}",
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        load_best_model_at_end=True,
        metric_for_best_model="eval_NanoBEIR_R100_mean_ndcg@10",
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        logging_steps=250,
        logging_first_step=True,
        run_name=run_name,  # Will be used in W&B if `wandb` is installed
        seed=12,
        report_to="none",
    )

    # 커스텀 데이터 콜레이터 생성
    data_collator = PrerankerDataCollator(
        tokenize_fn=lambda texts: model.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    )

    # 6. 커스텀 트레이너 생성 및 training 시작
    trainer = PrerankerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
        data_collator=data_collator,
    )
    trainer.train()

    # 7. Evaluate the final model, useful to include these in the model card
    evaluator(model)

    # 8. Save the final model
    final_output_dir = f"models/{run_name}/final"
    model.save_pretrained(final_output_dir)

    # 9. (Optional) save the model to the Hugging Face Hub!
    # It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
    try:
        model.push_to_hub(run_name)
    except Exception:
        logging.error(
            f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
            f"`huggingface-cli login`, followed by loading the model using `model = CrossEncoder({final_output_dir!r})` "
            f"and saving it using `model.push_to_hub('{run_name}')`."
        )


if __name__ == "__main__":
    main()
