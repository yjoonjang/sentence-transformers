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
        features, labels = self.collect_features(inputs)
        loss_fn = self.loss
        
        # bio_labels 추출
        bio_labels = inputs.get("bio_labels", None)

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