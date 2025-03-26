import logging
import torch
import torch.nn as nn
import os
from typing import List, Dict, Union, Tuple, Callable, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.autonotebook import trange

logger = logging.getLogger(__name__)

class CustomCrossEncoder:
    """
    A wrapper class for BGE Reranker in CrossEncoder format with additional bio classifier
    
    Args:
        model_name_or_path (str): Path to the model or model name from huggingface.co/models
        num_labels (int, optional): Number of output labels. Set to 2 for binary classification.
        max_length (int, optional): Maximum sequence length
        device (str, optional): Device to use for computation ("cuda", "cpu", etc.)
        use_fp16 (bool, optional): Whether to use half-precision floating point
        bio_classes (int, optional): Number of bio classifier classes
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int = 2,
        max_length: int = 8194,
        device: Optional[str] = None,
        use_fp16: bool = False,
        bio_classes: int = 11,
        **kwargs
    ):
        self.model_name_or_path = model_name_or_path
        self.num_labels = num_labels
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.bio_classes = bio_classes
        
        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, 
            num_labels=num_labels,
            output_hidden_states=True,  # Ensure we get hidden states
            **kwargs
        )
        
        # Look for saved bio_classifier weights file
        bio_weights_path = self._find_bio_weights_file(model_name_or_path)
        
        # Add bio_classifier regardless of whether it was in the loaded model
        logger.info(f"Adding bio_classifier with {bio_classes} classes to the model")
        self.model.bio_classifier = nn.Sequential(
            nn.Linear(1024, bio_classes, bias=True),
            nn.Softmax(dim=-1)
        )
        
        # Initialize bio_classifier weights
        if bio_weights_path:
            logger.info(f"Loading bio_classifier weights from {bio_weights_path}")
            try:
                bio_weights = torch.load(bio_weights_path, map_location=device)
                self.model.bio_classifier[0].weight = nn.Parameter(bio_weights["weight"])
                self.model.bio_classifier[0].bias = nn.Parameter(bio_weights["bias"])
                logger.info("Successfully loaded bio_classifier weights")
            except Exception as e:
                logger.error(f"Error loading bio_classifier weights: {str(e)}")
                logger.info("Initializing bio_classifier with random weights instead")
                self._initialize_random_bio_weights(bio_classes)
        else:
            logger.info("No bio_classifier weights found, initializing with random weights")
            self._initialize_random_bio_weights(bio_classes)
        
        # Move model to the specified device
        self.model.to(self.device)
        if self.use_fp16:
            self.model.half()
        
        # Set default activation function
        if num_labels == 1:
            self.activation_fn = nn.Sigmoid()
        elif num_labels == 2:
            # Use softmax for binary classification
            self.activation_fn = nn.Softmax(dim=1)
        else:
            self.activation_fn = nn.Identity()
    
    def _find_bio_weights_file(self, model_path):
        """
        Look for bio_classifier_weights.pt file in various potential locations
        
        Args:
            model_path: The model path or name
            
        Returns:
            Path to bio weights file if found, None otherwise
        """
        # Direct path check
        direct_path = os.path.join(model_path, "bio_classifier_weights.pt")
        if os.path.exists(direct_path):
            return direct_path
            
        # Check in Hugging Face cache
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
            try:
                # Try to download just the bio_classifier_weights.pt file
                cache_path = hf_hub_download(
                    repo_id=model_path,
                    filename="bio_classifier_weights.pt",
                    use_auth_token=None,  # Add token if needed for private repos
                )
                if os.path.exists(cache_path):
                    return cache_path
            except Exception as e:
                logger.info(f"Could not download bio_classifier_weights.pt directly: {str(e)}")
                
                # Try to download the full repo
                try:
                    cache_dir = snapshot_download(
                        repo_id=model_path,
                        use_auth_token=None,  # Add token if needed for private repos
                    )
                    cache_path = os.path.join(cache_dir, "bio_classifier_weights.pt")
                    if os.path.exists(cache_path):
                        return cache_path
                except Exception as e2:
                    logger.info(f"Could not download full repo: {str(e2)}")
        except ImportError:
            logger.info("huggingface_hub not available for downloading bio weights")
        
        # For cached models, try to find in the model's subfolder structure
        if not os.path.isdir(model_path):
            try:
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(model_path)
                if hasattr(config, "_name_or_path"):
                    alt_path = config._name_or_path
                    if os.path.isdir(alt_path):
                        alt_weights_path = os.path.join(alt_path, "bio_classifier_weights.pt")
                        if os.path.exists(alt_weights_path):
                            return alt_weights_path
            except Exception as e:
                logger.info(f"Error checking alternative paths: {str(e)}")
        
        return None
    
    def _initialize_random_bio_weights(self, bio_classes):
        """Initialize bio_classifier with random weights"""
        self.model.bio_classifier[0].weight = nn.Parameter(torch.randn(bio_classes, 1024))
        self.model.bio_classifier[0].bias = nn.Parameter(torch.randn(bio_classes))
    
    def predict(
        self,
        sentences: Union[List[Tuple[str, str]], List[List[str]], Tuple[str, str]],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        convert_to_numpy: bool = True,
        apply_softmax: bool = True,
        normalize_scores: bool = True,
        return_class_id: bool = True,
    ):
        """
        Predicts scores for the given sentence pairs.
        
        Args:
            sentences: List of sentence pairs [(sent1, sent2), ...] or a single pair (sent1, sent2)
            batch_size: Batch size for processing
            show_progress_bar: Whether to show a progress bar
            convert_to_numpy: Whether to convert the results to numpy arrays
            apply_softmax: Whether to apply softmax to logits (for num_labels > 1)
            normalize_scores: Whether to normalize scores
            return_class_id: Whether to return class IDs along with scores
            
        Returns:
            If return_class_id is False: prediction scores or class probabilities
            If return_class_id is True: tuple of (prediction scores, predicted class IDs)
        """
        input_was_string = False
        if isinstance(sentences, tuple) and len(sentences) == 2 and isinstance(sentences[0], str) and isinstance(sentences[1], str):
            sentences = [sentences]
            input_was_string = True
            
        if show_progress_bar is None:
            show_progress_bar = logger.level <= logging.INFO
            
        self.model.eval()
        all_scores = []
        all_class_ids = [] if return_class_id else None
        
        for start_idx in trange(0, len(sentences), batch_size, desc="Predicting", disable=not show_progress_bar):
            batch = sentences[start_idx:start_idx+batch_size]
            
            # Tokenize
            features = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            features = {k: v.to(self.device) for k, v in features.items()}
            
            with torch.no_grad():
                outputs = self.model(**features, return_dict=True)
                logits = outputs.logits
                
                if self.num_labels == 1:
                    scores = logits.squeeze(-1)
                    if return_class_id:
                        # Using argmax (value > 0 means class 1, else class 0)
                        class_ids = (scores > 0).long()
                else:
                    if apply_softmax:
                        probs = torch.softmax(logits, dim=1)
                    else:
                        probs = logits
                        
                    if return_class_id:
                        # Always use argmax instead of threshold
                        class_ids = torch.argmax(logits, dim=1)
                    
                    if self.num_labels == 2 and not normalize_scores:
                        scores = probs[:, 1]
                    else:
                        scores = probs
                
                if normalize_scores and self.num_labels == 2:
                    scores = probs[:, 1]
                
                all_scores.extend(scores.cpu())
                if return_class_id:
                    all_class_ids.extend(class_ids.cpu())
        
        if convert_to_numpy:
            all_scores = torch.stack(all_scores).numpy()
        else:
            all_scores = torch.stack(all_scores)
            
        # Convert class IDs
        if return_class_id:
            if convert_to_numpy:
                all_class_ids = torch.stack(all_class_ids).numpy()
            else:
                all_class_ids = torch.stack(all_class_ids)
        
        # Return only first result for single input
        if input_was_string:
            all_scores = all_scores[0]
            if return_class_id:
                all_class_ids = all_class_ids[0]
        
        # Return results
        if return_class_id:
            return all_scores, all_class_ids
        else:
            return all_scores
    
    def predict_token_bio(
        self,
        sentences: Union[List[Tuple[str, str]], List[List[str]], Tuple[str, str]],
        batch_size: int = 32,
        show_progress_bar: bool = None,
        convert_to_numpy: bool = True,
        return_class_id: bool = True,
    ):
        """
        Predicts bio classes for each token in the query part (before [SEP]) of the given sentence pairs.
        
        Args:
            sentences: List of sentence pairs [(sent1, sent2), ...] or a single pair (sent1, sent2)
            batch_size: Batch size for processing
            show_progress_bar: Whether to show a progress bar
            convert_to_numpy: Whether to convert the results to numpy arrays
            return_class_id: Whether to return class IDs along with probabilities
            
        Returns:
            A dictionary containing:
            - 'tokens': List of tokens for each query
            - 'probabilities': Token-level bio class probabilities
            - 'class_ids': (Optional) Predicted class for each token
        """
        input_was_string = False
        if isinstance(sentences, tuple) and len(sentences) == 2 and isinstance(sentences[0], str) and isinstance(sentences[1], str):
            sentences = [sentences]
            input_was_string = True
            
        if show_progress_bar is None:
            show_progress_bar = logger.level <= logging.INFO
            
        self.model.eval()
        
        # Lists to store results for each example
        all_tokens = []
        all_token_probs = []
        all_token_class_ids = [] if return_class_id else None
        
        for start_idx in trange(0, len(sentences), batch_size, desc="Token Bio Predicting", disable=not show_progress_bar):
            batch = sentences[start_idx:start_idx+batch_size]
            
            # Store tokens and their indices for current batch
            batch_tokens = []
            batch_query_indices = []
            
            # First, tokenize just the queries to get their length
            query_only_tokens = []
            for sent1, sent2 in batch:
                # Tokenize only the query part
                query_tokens = self.tokenizer.tokenize(sent1)
                query_only_tokens.append(query_tokens)
                
            # Now tokenize the full pairs
            features = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                return_token_type_ids=True,  # Get token type IDs
                add_special_tokens=True
            )
            
            # Identify query tokens using knowledge of query lengths
            for i in range(len(batch)):
                # The first token is [CLS], then come the query tokens, then [SEP]
                # So query tokens are at positions 1 to 1+len(query_tokens)
                query_len = len(query_only_tokens[i])
                
                # Add 1 for [CLS] token at the beginning - CLS는 일단 제외
                query_indices = list(range(1, 1 + query_len))
                
                batch_query_indices.append(query_indices)
                
                # Get tokens for visualization
                tokens = self.tokenizer.convert_ids_to_tokens(features.input_ids[i])
                query_tokens = [tokens[idx] for idx in query_indices]
                batch_tokens.append(query_tokens)
                
                # Debugging output
                logger.debug(f"Query: {batch[i][0]}")
                logger.debug(f"Query indices: {query_indices}")
                logger.debug(f"Query tokens: {query_tokens}")
                
            # Move tensors to device (exclude non-model inputs)
            features_for_model = {k: v.to(self.device) for k, v in features.items() 
                                if k not in ["offset_mapping", "token_type_ids"]}
            
            with torch.no_grad():
                # Run through base model to get hidden states
                outputs = self.model(**features_for_model, output_hidden_states=True, return_dict=True)
                
                # Get the last hidden states for all tokens
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    last_hidden_states = outputs.hidden_states[-1]
                else:
                    # Fallback to direct roberta module access
                    last_hidden_states = self.model.roberta(**features_for_model, output_hidden_states=True).hidden_states[-1]
                
                # Process each example in the batch
                for i in range(len(batch)):
                    # Get query token indices for this example
                    query_indices = batch_query_indices[i]
                    
                    # Get hidden states for only query tokens
                    query_hidden_states = last_hidden_states[i, query_indices]
                    
                    # Apply bio_classifier to each token's hidden state
                    token_bio_probs = self.model.bio_classifier(query_hidden_states)
                    
                    # Store results
                    all_tokens.append(batch_tokens[i])
                    all_token_probs.append(token_bio_probs.cpu())
                    
                    if return_class_id:
                        # Get predicted class for each token using argmax
                        token_class_ids = torch.argmax(token_bio_probs, dim=1)
                        all_token_class_ids.append(token_class_ids.cpu())
        
        # Convert to numpy if requested
        if convert_to_numpy:
            all_token_probs = [probs.numpy() for probs in all_token_probs]
            if return_class_id:
                all_token_class_ids = [ids.numpy() for ids in all_token_class_ids]
        
        # Handle single input case
        if input_was_string:
            result = {
                'tokens': all_tokens[0],
                'probabilities': all_token_probs[0]
            }
            if return_class_id:
                result['class_ids'] = all_token_class_ids[0]
        else:
            result = {
                'tokens': all_tokens,
                'probabilities': all_token_probs
            }
            if return_class_id:
                result['class_ids'] = all_token_class_ids
                
        return result
    
    def rank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        return_documents: bool = False,
        batch_size: int = 32,
        show_progress_bar: bool = None,
    ) -> List[Dict]:
        """
        Ranks a list of documents for a given query.
        
        Args:
            query: Query string
            documents: List of documents
            top_k: Number of top documents to return (None returns all)
            return_documents: Whether to include document text in results
            batch_size: Batch size
            show_progress_bar: Whether to show progress bar
            
        Returns:
            List of ranked documents (each item is a dict with {'corpus_id': index, 'score': score})
        """
        query_doc_pairs = [(query, doc) for doc in documents]
        
        scores = self.predict(
            query_doc_pairs,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            return_class_id=False
        )
        
        results = []
        for i, score in enumerate(scores):
            result = {
                "corpus_id": i,
                "score": float(score)
            }
            if return_documents:
                result["text"] = documents[i]
            results.append(result)
        
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        
        if top_k is not None:
            results = results[:top_k]
            
        return results
    
    def save(self, path: str):
        """
        Saves the model, tokenizer, and bio_classifier weights.
        
        Args:
            path: Path to save the model
        """
        os.makedirs(path, exist_ok=True)
        
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        bio_weights = {
            "weight": self.model.bio_classifier[0].weight.data,
            "bias": self.model.bio_classifier[0].bias.data
        }
        bio_weights_path = os.path.join(path, "bio_classifier_weights.pt")
        torch.save(bio_weights, bio_weights_path)
        
        logger.info(f"Model and bio_classifier saved to {path}")
        logger.info(f"Bio classifier weights saved separately to {bio_weights_path}")
    
    def to(self, device: str):
        """
        Moves the model to the specified device.
        
        Args:
            device: Target device ("cuda", "cpu", etc.)
        """
        self.device = device
        self.model.to(device)
        return self