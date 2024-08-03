from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import torch
from datasets import load_metric, load_from_disk, concatenate_datasets
import numpy as np
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments, Trainer
import argparse
import pathlib
import os
from evaluate import load


### FIND LEAF DIRECTORIES
def list_leaf_dirs(root_dir: pathlib.Path):
  root_dir = pathlib.Path(root_dir)
  leaf_dirs = []
  for path in root_dir.rglob("*"):
    if path.is_dir():
      is_leaf = True
      for i in path.iterdir():
        if i.is_dir():
          is_leaf = False
          break
      if is_leaf:
        leaf_dirs.append(path)
  return leaf_dirs

### PREPARE TRAINING
@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    print(pred_str[0:5])
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    print(label_str[0:5])
    wer_score = wer.compute(predictions=pred_str, references=label_str)

    return {"wer": wer_score}

### GET CHECKPOINT
def get_num(file):
    num = 0
    for i in range(len(file)):
        if file[i] >='0' and file[i] <='9':
           num = num * 10 + int(file[i])
    return num 
def get_checkpoint(directory):
    max_num = 0
    for file_name in os.listdir(directory):
        if 'checkpoint' in file_name:
            num = get_num(file_name)
            if num > max_num:
                max_num = num
    return max_num



tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
wer = load("wer")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir',type=str, required=True)
    parser.add_argument('--output_dir',type=str, required=True)
    parser.add_argument('--num_epochs',type=int, required=True)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--logging_steps', type=int, default = 100)

    args = parser.parse_args()
    
    input_dir = args.input_dir
    out_dir = args.output_dir
    epoch = args.num_epochs
    learning_rate = args.learning_rate
    batch_size = args.batch
    log = args.logging_steps
    
    dataset_paths = list_leaf_dirs(input_dir)
    datasets = [load_from_disk(str(path)) for path in dataset_paths]
    dataset = concatenate_datasets(datasets)
    eval_dataset = load_from_disk(str(dataset_paths[0]))
    print(str(dataset_paths[0]))

    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base",
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )

    model.gradient_checkpointing_enable()
    model.freeze_feature_encoder()

    training_args = TrainingArguments(
        output_dir=out_dir,
        group_by_length=True,
        per_device_train_batch_size=batch_size,
        eval_strategy="steps",
        num_train_epochs=epoch,
        fp16=True,
        gradient_checkpointing=True,
        save_steps=0.1,
        eval_steps=0.1,
        logging_steps=log,
        learning_rate=learning_rate,
        weight_decay=0.005,
        warmup_steps=500,
        save_total_limit=10,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,
    )

    print('Start training:')
    trainer.train()

    checkpoint = get_checkpoint(out_dir)
    checkpoint_path = os.path.join(out_dir,'checkpoint-'+str(checkpoint))
    tokenizer.save_pretrained(checkpoint_path)