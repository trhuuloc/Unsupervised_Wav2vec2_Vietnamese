import pathlib
from datasets import load_dataset
import os
import argparse
import re
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
import psutil
import sys
import shutil
import stat
import tempfile

    
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

def extract_path(path, need):
    # print(need)
    subpath = path.parts[path.parts.index(need) + 1:]
    return pathlib.Path(*subpath)


### NORMALIZE
chars_to_ignore_regex = r"[\,'\?\.\!\-\;\:\"0-9fjwzFJWZ]"

def remove_special_characters(batch):
    batch[labels_name] = re.sub(chars_to_ignore_regex, '', batch[labels_name]).lower()
    return batch



def prepare_dataset(batch):
    audio = batch[audio_name]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]

    with processor.as_target_processor():
        batch["labels"] = processor(batch[labels_name]).input_ids
    return batch
    


# print(a)
def avoid_OOM(percent=60):
  memo = psutil.virtual_memory()
  if int(memo[2]) > percent:
    print(f"Memory reached {memo[2]}%, terminate to avoid out of memory")
    sys.exit(0)
    
def clear_audio_folder_cache(path):
  shutil.rmtree(pathlib.Path(path), ignore_errors=False)
  print(f'Cleared audio folder cache at {path}')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir',type=str, required=True)
    parser.add_argument('--output_dir',type=str, required=True)
    parser.add_argument('--cache_dir',type=str, required=True)
    parser.add_argument('--labels_column_name',type=str, default="transcription")
    parser.add_argument('--audio_column_name',type=str, default="audio")

    args = parser.parse_args()
    
    input_dir = args.input_dir
    out_dir = args.output_dir
    labels_name = args.labels_column_name
    audio_name = args.audio_column_name
    cache_dir = args.cache_dir
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir) 

    a = list_leaf_dirs(input_dir)

    tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    for i in a:
        dataset = load_dataset("audiofolder", data_dir=i, split="train",trust_remote_code=True, cache_dir=cache_dir)
        dataset = dataset.map(remove_special_characters)    
        avoid_OOM(70)
        dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names, num_proc=1)

        path = extract_path(i, input_dir)
        dataset.save_to_disk(os.path.join(out_dir,path))
            