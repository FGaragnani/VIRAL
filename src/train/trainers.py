from dataclasses import dataclass
from transformers import (
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    BaseImageProcessor,
    FeatureExtractionMixin,
    ProcessorMixin,
    EvalPrediction,
    DataCollator,
    TrainerCallback
)
from typing import Any, Callable, Optional, Union
import src.custom_utils as utils
from torch.utils.data import DataLoader
import webdataset as wds
from src.train.args import DataArguments
from transformers.utils.deprecation import deprecate_kwarg
import torch.nn as nn
import torch
from torch.utils.data import Dataset, IterableDataset

logger = utils.get_logger()


@dataclass
class CustomTrainer(Trainer):

    @deprecate_kwarg("tokenizer", new_name="processing_class", version="5.0.0", raise_if_both_names=True)
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, None] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset], "datasets.Dataset"]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        optimizer_cls_and_kwargs: Optional[tuple[type[torch.optim.Optimizer], dict[str, Any]]] = None,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        data_args: Optional[DataArguments] = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )
        self.data_args = data_args

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        super().save_model(output_dir, _internal_call)
        self.processing_class.save_pretrained(output_dir)

    def get_train_dataloader(self) -> DataLoader:        
        if isinstance(self.train_dataset, wds.DataPipeline):
            logger.info("Using WebDataset for training")
            return wds.WebLoader(
                self.train_dataset,
                batch_size=None,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                shuffle=False, # assuming shuffling is handled by WebDataset,
                collate_fn=self.data_collator,
                persistent_workers=self.args.dataloader_persistent_workers,
            )
        else:
            return super().get_train_dataloader()
    