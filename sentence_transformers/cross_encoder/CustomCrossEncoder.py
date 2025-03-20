import logging
import torch
import torch.nn as nn
from typing import List, Dict, Union, Tuple, Callable, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.autonotebook import trange

logger = logging.getLogger(__name__)

class CustomCrossEncoder:
    """
    BGE Reranker를 CrossEncoder 형식으로 래핑한 클래스
    
    Args:
        model_name_or_path (str): 모델 이름 또는 경로 (e.g., "BAAI/bge-reranker-v2-m3")
        num_labels (int, optional): 출력 레이블 수. 2로 설정하면 이진 분류기가 됩니다.
        max_length (int, optional): 최대 시퀀스 길이
        device (str, optional): 계산에 사용할 디바이스 ("cuda", "cpu" 등)
        use_fp16 (bool, optional): 반정밀도(FP16) 사용 여부
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int = 2,
        max_length: int = 512,
        device: Optional[str] = None,
        use_fp16: bool = False,
        **kwargs
    ):
        self.model_name_or_path = model_name_or_path
        self.num_labels = num_labels
        self.max_length = max_length
        self.use_fp16 = use_fp16
        
        # 디바이스 설정
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # 토크나이저 및 모델 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, 
            num_labels=num_labels,
            **kwargs
        )
        
        # 모델을 지정된 디바이스로 이동
        self.model.to(self.device)
        if self.use_fp16:
            self.model.half()
        
        # 기본 활성화 함수 설정
        if num_labels == 1:
            self.activation_fn = nn.Sigmoid()
        elif num_labels == 2:
            # 이진 분류의 경우 softmax 사용
            self.activation_fn = nn.Softmax(dim=1)
        else:
            self.activation_fn = nn.Identity()
    
    def predict(
        self,
        sentences: Union[List[Tuple[str, str]], List[List[str]], Tuple[str, str]],
        batch_size: int = 32,
        show_progress_bar: bool = None,
        convert_to_numpy: bool = True,
        apply_softmax: bool = True,
        normalize_scores: bool = True,
        return_class_id: bool = False,  # 클래스 ID 반환 여부
    ):
        """
        주어진 문장 쌍에 대한 점수를 예측합니다.
        
        Args:
            sentences: 문장 쌍 리스트 [(문장1, 문장2), ...] 또는 단일 문장 쌍 (문장1, 문장2)
            batch_size: 배치 크기
            show_progress_bar: 진행 바 표시 여부
            convert_to_numpy: 결과를 numpy 배열로 변환할지 여부
            apply_softmax: 로짓에 softmax를 적용할지 여부 (num_labels > 1인 경우)
            normalize_scores: 점수를 정규화할지 여부
            return_class_id: argmax를 사용해 클래스 ID도 함께 반환할지 여부
            
        Returns:
            return_class_id가 False인 경우: 예측 점수 또는 예측 클래스 확률
            return_class_id가 True인 경우: (예측 점수, 예측된 클래스 ID(0 또는 1)) 튜플
        """
        input_was_string = False
        if isinstance(sentences, tuple) and len(sentences) == 2 and isinstance(sentences[0], str) and isinstance(sentences[1], str):
            sentences = [sentences]
            input_was_string = True
            
        if show_progress_bar is None:
            show_progress_bar = logger.level <= logging.INFO
            
        self.model.eval()
        all_scores = []
        all_class_ids = [] if return_class_id else None  # 클래스 ID를 저장할 리스트
        
        for start_idx in trange(0, len(sentences), batch_size, desc="Predicting", disable=not show_progress_bar):
            batch = sentences[start_idx:start_idx+batch_size]
            
            # 토큰화
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
                        class_ids = (scores > 0.5).long()
                else:
                    if apply_softmax:
                        probs = torch.softmax(logits, dim=1)
                    else:
                        probs = logits
                        
                    if return_class_id:
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
            
        # 클래스 ID도 변환
        if return_class_id:
            if convert_to_numpy:
                all_class_ids = torch.stack(all_class_ids).numpy()
            else:
                all_class_ids = torch.stack(all_class_ids)
        
        # 단일 입력인 경우 첫 번째 결과만 반환
        if input_was_string:
            all_scores = all_scores[0]
            if return_class_id:
                all_class_ids = all_class_ids[0]
        
        # 결과 반환 (점수만 또는 점수와 클래스 ID)
        if return_class_id:
            return all_scores, all_class_ids
        else:
            return all_scores
    
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
        쿼리에 대해 문서 목록을 순위화합니다.
        
        Args:
            query: 쿼리 문자열
            documents: 문서 목록
            top_k: 반환할 상위 문서 수 (None이면 모두 반환)
            return_documents: 결과에 문서 텍스트 포함 여부
            batch_size: 배치 크기
            show_progress_bar: 진행 바 표시 여부
            
        Returns:
            순위화된 문서 리스트 (각 항목은 {'corpus_id': 인덱스, 'score': 점수} 형식)
        """
        # 쿼리-문서 쌍 생성
        query_doc_pairs = [(query, doc) for doc in documents]
        
        # 점수 예측
        scores = self.predict(
            query_doc_pairs,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True
        )
        
        # 결과 생성
        results = []
        for i, score in enumerate(scores):
            result = {
                "corpus_id": i,
                "score": float(score)
            }
            if return_documents:
                result["text"] = documents[i]
            results.append(result)
        
        # 점수 기준 내림차순 정렬
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        
        # top_k 반환
        if top_k is not None:
            results = results[:top_k]
            
        return results

    def predict_with_details(
        self,
        sentences: Union[List[Tuple[str, str]], List[List[str]], Tuple[str, str]],
        batch_size: int = 32,
        show_progress_bar: bool = None,
    ):
        """
        주어진 문장 쌍에 대한 자세한 예측 정보를 반환합니다.
        
        Args:
            sentences: 문장 쌍 리스트 [(문장1, 문장2), ...] 또는 단일 문장 쌍 (문장1, 문장2)
            batch_size: 배치 크기
            show_progress_bar: 진행 바 표시 여부
            
        Returns:
            단일 입력인 경우: 딕셔너리 (예측 클래스, 점수, 확률 등 포함)
            다중 입력인 경우: 딕셔너리 리스트
        """
        input_was_string = False
        if isinstance(sentences, tuple) and len(sentences) == 2 and isinstance(sentences[0], str) and isinstance(sentences[1], str):
            sentences = [sentences]
            input_was_string = True
            
        if show_progress_bar is None:
            show_progress_bar = logger.level <= logging.INFO
            
        self.model.eval()
        all_results = []
        
        for start_idx in trange(0, len(sentences), batch_size, desc="Predicting", disable=not show_progress_bar):
            batch = sentences[start_idx:start_idx+batch_size]
            
            # 토큰화
            features = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # 모델에 입력 전달
            features = {k: v.to(self.device) for k, v in features.items()}
            
            with torch.no_grad():
                outputs = self.model(**features, return_dict=True)
                logits = outputs.logits
                
                for i in range(len(batch)):
                    batch_logits = logits[i]
                    
                    if self.num_labels == 1:
                        # 회귀 모델
                        score = batch_logits.item()
                        predicted_class = 1 if score > 0.5 else 0
                        confidence = score if predicted_class == 1 else 1 - score
                        
                        result = {
                            "score": score,
                            "prediction_class": predicted_class,
                            "confidence": confidence
                        }
                    else:
                        # 분류 모델
                        probs = torch.softmax(batch_logits, dim=0)
                        predicted_class = torch.argmax(batch_logits).item()
                        confidence = probs[predicted_class].item()
                        
                        result = {
                            "raw_logits": batch_logits.cpu().numpy().tolist(),
                            "probabilities": probs.cpu().numpy().tolist(),
                            "prediction_class": predicted_class,
                            "confidence": confidence,
                            "positive_class_prob": probs[1].item() if self.num_labels == 2 else None
                        }
                    
                    all_results.append(result)
        
        # 단일 입력인 경우 첫 번째 결과만 반환
        if input_was_string:
            return all_results[0]
        
        return all_results
    
    def save(self, path: str):
        """
        모델과 토크나이저를 저장합니다.
        
        Args:
            path: 저장 경로
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
    def to(self, device: str):
        """
        모델을 특정 디바이스로 이동합니다.
        
        Args:
            device: 타겟 디바이스 ("cuda", "cpu" 등)
        """
        self.device = device
        self.model.to(device)
        return self