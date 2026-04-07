"""
Copyright 2024 LINE Corporation
LINE Corporation licenses this file to you under the Apache License,
version 2.0 (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at:
    https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.
"""

import os
import codecs as cs
import orjson  # loading faster than json
import json

import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import random
import torch.nn.functional as F


def token_process(token_num: int, name: str, text:torch.Tensor, tokenizer):
    
    if name == 'ViT-B/32':
        if isinstance(text, str):
            text = [text]
        context_length = token_num
        truncate = True
        sot_token = tokenizer.encoder["<|startoftext|>"]
        eot_token = tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + tokenizer.encode(txt) + [eot_token] for txt in text]
        texts_token = torch.zeros(len(all_tokens), context_length, dtype=torch.int)
        length = torch.zeros(len(all_tokens), dtype=torch.int)
        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate:
                    tokens = tokens[:context_length]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f"Input {text[i]} is too long for context length {context_length}")
            texts_token[i, :len(tokens)] = torch.tensor(tokens)
            length[i] = len(tokens)
        return texts_token, length
    else:
        texts_token = tokenizer(
            text, padding='max_length',max_length=token_num, truncation=True, return_tensors="pt"
        )
        length = texts_token.attention_mask.to(dtype=bool).sum(1)
        return texts_token.input_ids, length
    
def sentence_process(texts:torch.Tensor, device, sentence_tokenizer,sentence_text_model):
    sentences_tokens = sentence_tokenizer(
        texts, padding=True, truncation=True, return_tensors="pt"
    )
    output = sentence_text_model(**sentences_tokens.to(device))
    attention_mask = sentences_tokens["attention_mask"]

    token_embeddings = output["last_hidden_state"]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sentence_embeddings = torch.sum(
        token_embeddings * input_mask_expanded, 1
    ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings



def read_split(path, split):
    split_file = os.path.join(path, "splits", split + ".txt")
    id_list = []
    with cs.open(split_file, "r") as f:
        for line in f.readlines():
            id_list.append(line.strip())
    return id_list

def load_annotations(path, name="annotations.json"):
    json_path = os.path.join(path, name)
    with open(json_path, "rb") as ff:
        return orjson.loads(ff.read())
    
def load_events(path, annotations, keyids, mean, std, fps=20):
    events = {}
    eps = 1e-12
    for keyid in tqdm(keyids):
        
        annotation = annotations[keyid]
        #load motion
        m_path = annotation["path"]
        start = annotation["annotations"][0]["start"]
        end = annotation["annotations"][0]["end"]
        begin = int(start * fps)
        end = int(end * fps)
        
        motion_path = os.path.join(path, 'motions/guoh3dfeats', m_path + ".npy")
        motion = np.load(motion_path)
        motion = torch.from_numpy(motion).to(torch.float)
        #normalize here
        motion = (motion - mean) / (std + eps)
        motion = motion[begin:end]

        #load events
        idx = 0
        event_texts = {}
        for item in range(len(annotation["annotations"])):
            name = str(keyid) + "_" + str(idx) + ".json"
            json_path = os.path.join(path, 'event_texts', name)
            with open(json_path, "rb") as ff:
                event_file = orjson.loads(ff.read())
                event_texts[idx] = event_file["events"]
            idx = idx + 1
        events[keyid]={
            "event": event_texts,
            "motion": motion,
            "length": len(motion)
        }
        
                
            
    return events

class TextMotionDataset(Dataset):
    def __init__(
        self,
        cfg,
        split,
        min_seconds: float = 2.0,
        max_seconds: float = 10.0,
        preload: bool = False,
        tiny: bool = False
    ):
        if tiny:
            split = split + "_tiny"
        
        self.cfg = cfg
        self.split = split
        self.keyids = read_split(cfg.dataset.data_dir, split)

        self.min_seconds = min_seconds
        self.max_seconds = max_seconds
        self.max_motion_length = cfg.dataset.max_motion_length
        self.ev2ev = cfg.dataset.ev2ev

        self.mean = torch.load(cfg.mean_path)
        self.std = torch.load(cfg.std_path)

        # remove too short or too long annotations
        self.annotations = load_annotations(cfg.dataset.data_dir)
        
        self.events = load_events(cfg.dataset.data_dir, self.annotations, self.keyids, self.mean, self.std)
        # filter annotations (min/max)
        # but not for the test set
        # otherwise it is not fair for everyone
        if "test" not in split:
            self.annotations = self.filter_annotations(self.annotations)


        self.is_training = split == "train"
        
        self.keyids = [keyid for keyid in self.keyids if keyid in self.annotations]
        
        if preload:
            for _ in tqdm(self, desc="Preloading the dataset"):
                continue

    def __len__(self):
        return len(self.keyids)

    def load_keyid(self, keyid):
        annotations = self.annotations[keyid]
        events = self.events[keyid]
        index = 0
        if self.is_training:
            index = np.random.randint(len(annotations["annotations"]))
        annotation = annotations["annotations"][index]
        # check if it is orig->event or event->event
        event = events["event"][index]
        
        if self.ev2ev:
            text = ''
            #replace articles
            for clause in event:
                text += clause + ' '
        else:
            text = annotation["text"]
    

        elen = len(event)
        e_len = np.array([elen])
        e_len = torch.tensor(e_len).int()
        m_length = events["length"]
        motion = events["motion"]
        
        #prepare negative samples already
        shuffled = sorted(event, key=lambda k: random.random())
        shuffled_event_text = ''
        for clause in shuffled:
            shuffled_event_text += clause + ' '
        max_motion_length = self.max_motion_length
        if m_length >= self.max_motion_length:
            idx = 0#random.randint(0, len(motion) - max_motion_length)
            motion = motion[0 : 0 + self.max_motion_length].float()
            m_length = max_motion_length
        else:
            if self.cfg.dataset.padding:
                padding_len = max_motion_length - m_length
                C = motion.shape[1]
                padding_zeros = np.zeros((padding_len, C), dtype=np.float32)
                motion = np.concatenate((motion, padding_zeros), axis=0)
                motion = torch.from_numpy(motion).float()
        
        
        return text, motion, m_length, e_len, shuffled_event_text, keyid

    def __getitem__(self, item):
        
        keyid = self.keyids[item]
        text, motion, m_length, e_len, shuffled_event_text, _ = self.load_keyid(keyid)
        
        return text, motion, m_length, e_len, shuffled_event_text, keyid
        

    def filter_annotations(self, annotations):
        filtered_annotations = {}
        for key, val in annotations.items():
            annots = val.pop("annotations")
            filtered_annots = []
            for annot in annots:
                duration = annot["end"] - annot["start"]
                if self.max_seconds >= duration >= self.min_seconds:
                    filtered_annots.append(annot)

            if filtered_annots:
                val["annotations"] = filtered_annots
                filtered_annotations[key] = val

        return filtered_annotations

