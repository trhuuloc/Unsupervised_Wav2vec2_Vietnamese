import os
import torch
import argparse
from transformers import Wav2Vec2ForCTC
from datasets import load_metric, load_dataset, Audio
import numpy as np
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
import re



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

### NORMALIZE
def prepare_dataset(batch):
    audio = batch[audio_name]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]

    return batch

### EVALUATION
chars_to_ignore_regex = r"[\,'\?\.\!\-\;\:\"0-9fjwzFJWZ]"
def map_to_result(batch):
  with torch.no_grad():
    input_values = torch.tensor(batch["input_values"]).unsqueeze(0)
    logits = model(input_values).logits

  pred_ids = torch.argmax(logits, dim=-1)
  batch["pred_str"] = processor.batch_decode(pred_ids)[0]
  batch["labels"] = re.sub(chars_to_ignore_regex, '', batch[labels_name]).lower()

  return batch


    
tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

wer_metric = load_metric("wer", trust_remote_code = True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir',type=str, required=True)
    parser.add_argument('--eval_data',type=str, required=True)

    args = parser.parse_args()
    
    input_dir = args.input_dir
    eval_data = args.eval_data
    
    if eval_data == "vivos":
        eval_dataset = load_dataset("AILAB-VNUHCM/vivos", trust_remote_code=True)
        eval_dataset = eval_dataset["test"]
        eval_dataset = eval_dataset.remove_columns(["speaker_id","path"])
        
        labels_name = "sentence"
        audio_name = "audio"
        eval_dataset = eval_dataset.map(prepare_dataset, num_proc=1)
    elif eval_data == "common-voice":
        eval_dataset = load_dataset("mozilla-foundation/common_voice_11_0", "vi", split="test", trust_remote_code=True)
        eval_dataset = eval_dataset.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
        
        labels_name = "sentence"
        audio_name = "audio"
        eval_dataset = eval_dataset.cast_column("audio", Audio(sampling_rate=16_000))
        eval_dataset = eval_dataset.map(prepare_dataset, num_proc=1)



checkpoint = get_checkpoint(input_dir)
checkpoint_path = os.path.join(input_dir,'checkpoint-'+str(checkpoint))

processor = Wav2Vec2Processor.from_pretrained(checkpoint_path)
model = Wav2Vec2ForCTC.from_pretrained(checkpoint_path)


results = eval_dataset.map(map_to_result, remove_columns=eval_dataset.column_names)
print("Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["labels"])))