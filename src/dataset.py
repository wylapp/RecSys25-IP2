import torch
import json
import pickle
from torch import nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
import os
from os import path
import random
from loguru import logger
from src import utils
from tqdm import tqdm
import pickle


def load_ent_embedding(dpath:str):
    embedding_grid = pd.read_table(path.join(dpath, "entity_embedding.vec"), sep='\t', header=None, )
    vocabulary = embedding_grid.iloc[:, 0]  # column 0: entity ID
    vectors = embedding_grid.iloc[:, 1:-1] # last column: nan!

    word_vectors = {}
    for i in range(len(vocabulary)):
        word = vocabulary[i]
        vector = vectors.iloc[i].values.astype(np.float32)
        word_vectors[word] = vector

    ent_map = { k:i for i, k in enumerate(word_vectors)}
    # entity ID-embedding index mapping dict
    return ent_map, word_vectors


class newsArchiveDataset(Dataset):

    def __init__(self, config, stage='train') -> None:
        super(newsArchiveDataset, self).__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_path)
        with open(os.path.join(config.data_dir, config.dataset, stage, "ent2id.pkl"), "rb") as pf:
            self.ent_map = pickle.load(pf)
        self.org_newsgrid = pd.read_table(path.join("data", config.dataset, stage, 'news.tsv'), sep='\t', header=None, names=[
                  'id', 'category', 'subcategory', 'title', 'abstract', 'url',
                  'title_entities', 'abstract_entities'
              ], quotechar=" ")
        self._preprocess()
        # TODO: implement processed data export and import.
        with open(os.path.join(config.data_dir, config.dataset, stage, "cat2id.pkl"), "rb") as pf:
            self.cat2id = pickle.load(pf)
        with open(os.path.join(config.data_dir, config.dataset, stage, "subcat2id.pkl"), "rb") as pf:
            self.subcat2id = pickle.load(pf)
    
    def _preprocess(self):
        # append original pandas.DataFrame
        self.org_newsgrid["title_ids"] = [[] for _ in range(len(self.org_newsgrid))]
        self.org_newsgrid["masks"] = [[] for _ in range(len(self.org_newsgrid))]
        # allocate columns

        for i, row in tqdm(self.org_newsgrid.iterrows()):
            ori_text = row['title']
            token_emb = self.tokenizer(ori_text, add_special_tokens=True, truncation=True,
                                       max_length=self.config.num_words_title, padding='max_length')
            self.org_newsgrid.at[i, "title_ids"] = token_emb["input_ids"]
            self.org_newsgrid.at[i, "masks"] = token_emb["attention_mask"]

        logger.info(f"Preprocess finished, data row {len(self.org_newsgrid)}")


    def __getitem__(self, index) -> list:
        def parse_jt(jt:str) -> list[list]:
            """Converting Json obj in Tsv file into python obj
            """
            embedding_ids = []
            for ent in json.loads(jt):
                if (eid := self.ent_map.get(ent["WikidataId"], None)) == None:
                    # logger.warning(f"Entity missing! {ent['WikidataId']}")
                    pass
                else:
                    embedding_ids.append(eid)
            return embedding_ids

        news_series = self.org_newsgrid.iloc[index, :]
        
        return [news_series.title_ids, news_series.masks, parse_jt(news_series.title_entities), 
                self.cat2id[news_series.category], self.subcat2id[news_series.subcategory]]

    def __len__(self) -> int:
        return self.org_newsgrid.shape[0]
    
    @property
    def sta_cat(self) -> int:
        return len(self.cat2id)
    
    @property
    def sta_subcat(self) -> int:
        return len(self.subcat2id)
    
    @property
    def sta_ent(self) -> int:
        return len(self.ent_map)
    
    @property
    def emb_size(self) -> int:
        return self.sta_cat + self.sta_subcat + self.sta_ent + 1


class preDataLoader(DataLoader):
    def __init__(self, data:newsArchiveDataset, config, shuffle=True):
        self.data = data
        self.batch_size = config.pretrain_batch
        # ! caution! pretrain batch and batch size are two different params
        self.shuffle = shuffle
        self.config = config
        self.indices = list(range(len(data)))
        if shuffle:
            random.shuffle(self.indices)
        self.batch_num = 0

    def __len__(self):
        return int(len(self.data) / float(self.batch_size))

    def num_batches(self):
        return len(self.data) / float(self.batch_size)
    
    def _process(self, idx_iter):
        tokens = []
        masks = []
        embeddings = []
        pos_index = []
        cat_ids = []
        subcat_ids = []
        for j, i in enumerate(idx_iter):
            news_entity = self.data[i]
            tokens.append(news_entity[0])
            masks.append(news_entity[1])
            if (ne_len:=len(news_entity[2])) != 0:
                embeddings.append([news_entity[3], news_entity[4]]+ news_entity[2])
            else:
                embeddings.append([news_entity[3], news_entity[4]])
            # batch-agnostic style!
            pos_index.extend([j]*1)
                # add a place-holder
            # will consider news without entities (by inserting zero)
            cat_ids.append(news_entity[3])
            subcat_ids.append(news_entity[4])
        return (tokens, masks, embeddings, pos_index, cat_ids, subcat_ids)

    def _cuda(self, data_items):
        return (torch.tensor(data_items[0], device=self.config.device), 
                torch.tensor(data_items[1], device=self.config.device),
                torch.nested.nested_tensor(data_items[2], device=self.config.device).to_padded_tensor(padding=0), 
                data_items[3],
                torch.tensor(data_items[4], device=self.config.device),
                torch.tensor(data_items[5], device=self.config.device))

    def __iter__(self):
        self.indices = list(range(len(self.data)))
        if self.shuffle:
            random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.indices != []:
            idxs = self.indices[:self.batch_size]
            batch = self._process(idxs)
            self.indices = self.indices[self.batch_size:]
            # * Only consider title and title-ents only
            if self.config.enable_gpu == True:
                batch = self._cuda(batch)
            return batch
        else:
            raise StopIteration

    def get(self):
        self.reset()
        return self.__next__()

    def reset(self):
        self.indices = list(range(len(self.data)))
        if self.shuffle: random.shuffle(self.indices)


class MindDataset(Dataset):
    def __init__(self, config, stage, news) -> None:
        super(MindDataset, self).__init__()
        self.config = config
        self.news_set = news
        self.stage = stage
        self.news_id = None
        richlist = os.listdir(path.join(config.root_data_dir, config.dataset, stage))
        if "behav_rich.tar.gz" in richlist and config.force_new != True:
            self.behaviors_parsed = pd.read_pickle(path.join(config.root_data_dir, config.dataset, stage, "behav_rich.tar.gz"))
            logger.success("Load rich-info behavior data!")
        else:
            self.behaviors_parsed = pd.read_table(path.join(config.root_data_dir, config.dataset, stage, "behavior_parsed.tsv"))
            self._preprocess()

    def token_bundle(self, id_list) -> list:
        for i, row in tqdm(id_list.iterrows()):
            token_lis = []
            mask_lis = []
            ent_lis = []
            ent_cat = []
            ent_subcat = []
            # ent_span = []
            # Do not need ent_span any more.
            for id in row["clicked_news"].split(" "):
                token_lis.append(self.news_set[self.news_id[id]][0])
                mask_lis.append(self.news_set[self.news_id[id]][1])
                if (ne_len:=len(self.news_set[self.news_id[id]][2])) != 0:
                    # only consider news with entities
                    ent_lis.append([self.news_set[self.news_id[id]][3], self.news_set[self.news_id[id]][4]]+self.news_set[self.news_id[id]][2])
                else:
                    ent_lis.append([self.news_set[self.news_id[id]][3], self.news_set[self.news_id[id]][4]])
                ent_cat.append(self.news_set[self.news_id[id]][3])
                ent_subcat.append(self.news_set[self.news_id[id]][4])
            id_list.at[i, "news_token"] = token_lis
            id_list.at[i, "news_mask"] = mask_lis
            id_list.at[i, "news_ent"] = ent_lis
            id_list.at[i, "news_cat"] = ent_cat
            id_list.at[i, "news_subcat"] = ent_subcat

            token_lis = []
            mask_lis = []
            ent_lis = []
            ent_cat = []
            ent_subcat = []
            for id in row["candidate_news"].split(" "):
                token_lis.append(self.news_set[self.news_id[id]][0])
                mask_lis.append(self.news_set[self.news_id[id]][1])
                if (ne_len:=len(self.news_set[self.news_id[id]][2])) != 0:

                    ent_lis.append([self.news_set[self.news_id[id]][3], self.news_set[self.news_id[id]][4]]+self.news_set[self.news_id[id]][2])
                else:
                    ent_lis.append([self.news_set[self.news_id[id]][3], self.news_set[self.news_id[id]][4]])
                ent_cat.append(self.news_set[self.news_id[id]][3])
                ent_subcat.append(self.news_set[self.news_id[id]][4])
            id_list.at[i, "impression_token"] = token_lis
            id_list.at[i, "impression_mask"] = mask_lis
            id_list.at[i, "impression_ent"] = ent_lis
            id_list.at[i, "impression_cat"] = ent_cat
            id_list.at[i, "impression_subcat"] = ent_subcat
        return id_list
    
    def _preprocess(self):   
        with open(path.join(self.config.root_data_dir, self.config.dataset, self.stage, "news2id.pkl"), "rb") as pf:
            self.news_id = pickle.load(pf)
        # * `newsArchiveDataset` have considered padding inside the news title tokens.
        self.behaviors_parsed["news_token"] = [[] for _ in range(len(self.behaviors_parsed))]
        self.behaviors_parsed["impression_token"] = [[] for _ in range(len(self.behaviors_parsed))]
        self.behaviors_parsed["news_mask"] = [[] for _ in range(len(self.behaviors_parsed))]
        self.behaviors_parsed["impression_mask"] = [[] for _ in range(len(self.behaviors_parsed))]
        self.behaviors_parsed["news_ent"] = [[] for _ in range(len(self.behaviors_parsed))]
        self.behaviors_parsed["impression_ent"] = [[] for _ in range(len(self.behaviors_parsed))]
        self.behaviors_parsed["news_cat"] = [[] for _ in range(len(self.behaviors_parsed))]
        self.behaviors_parsed["news_subcat"] = [[] for _ in range(len(self.behaviors_parsed))]
        self.behaviors_parsed["impression_cat"] = [[] for _ in range(len(self.behaviors_parsed))]
        self.behaviors_parsed["impression_subcat"] = [[] for _ in range(len(self.behaviors_parsed))]
        with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
            chunks = np.array_split(self.behaviors_parsed, self.config.num_workers)
            results = executor.map(self.token_bundle, chunks)
            self.behaviors_parsed = pd.concat(results)
        self.behaviors_parsed.to_pickle(path.join(self.config.root_data_dir, self.config.dataset, self.stage, "behav_rich.tar.gz"))
        
    
    def __len__(self) -> int:
        return len(self.behaviors_parsed)

    def __getitem__(self, index:int):
        bh_parsed = self.behaviors_parsed.iloc[index, :]
        return (bh_parsed["news_token"], bh_parsed["impression_token"], 
                bh_parsed["news_mask"], bh_parsed["impression_mask"], 
                bh_parsed["news_ent"], bh_parsed["impression_ent"], 
                bh_parsed["news_cat"], bh_parsed["impression_cat"],
                bh_parsed["news_subcat"], bh_parsed["impression_subcat"], 
                bh_parsed["clicked"])
        
class MINDloader(DataLoader):
    def __init__(self, data:MindDataset, config, shuffle=True, stage="train"):
        self.data = data
        self.batch_size = config.batch_size
        # ! caution! pretrain_batch and batch_size are two different params
        self.shuffle = shuffle
        self.config = config
        self.indices = list(range(len(data)))
        if shuffle:
            random.shuffle(self.indices)
        self.stage = stage
        self.batch_num = 0

    def __len__(self):
        return int(len(self.data) / float(self.batch_size))

    def num_batches(self):
        return len(self.data) / float(self.batch_size)
    
    def _process(self, idx_iter):
        hist_tokens = []
        hist_masks = []
        impress_tokens = []
        impress_masks = []
        length_record = []
        impress_length_record = []
        hist_ents = []
        impression_ents = []
        hist_cat = []
        hist_subcat = []
        impression_cat = []
        impression_subcat = []
        ground_truth = []
        for j, i in enumerate(idx_iter):
            news_entity = self.data[i]
            # history news tokens, sample news tokens, ground_truth
            hist_tokens.extend(news_entity[0])
            impress_tokens.extend(news_entity[1])
            hist_masks.extend(news_entity[2])
            impress_masks.extend(news_entity[3])
            length_record.append(len(news_entity[0]))
            impress_length_record.append(len(news_entity[1]))
            hist_cat.extend(news_entity[6])
            hist_subcat.extend(news_entity[8])
            impression_cat.extend(news_entity[7])
            impression_subcat.extend(news_entity[9])
            if self.stage == "train":
                ground_truth.append(0)
            else:
                ground_truth.append([int(x) for x in news_entity[-1].split(" ")])
            hist_ents.extend(news_entity[4])
            impression_ents.extend(news_entity[5])

        return [hist_tokens, impress_tokens, hist_masks, 
                impress_masks, length_record, impress_length_record, 
                hist_ents, impression_ents, hist_cat, impression_cat, hist_subcat, impression_subcat, ground_truth]

    def _cuda(self, data_items):
        data_items[6] = [torch.tensor(its) for its in data_items[6]]
        data_items[7] = [torch.tensor(its) for its in data_items[7]]
        return (torch.tensor(data_items[0], device=self.config.device),
                torch.tensor(data_items[1], device=self.config.device),
                torch.tensor(data_items[2], device=self.config.device),
                torch.tensor(data_items[3], device=self.config.device),
                data_items[4],
                torch.tensor(data_items[5], device=self.config.device, requires_grad=False),
                pad_sequence(data_items[6], batch_first=True, padding_value=0).to(self.config.device), 
                pad_sequence(data_items[7], batch_first=True, padding_value=0).to(self.config.device), 
                torch.tensor(data_items[8], device=self.config.device, requires_grad=False), 
                torch.tensor(data_items[9], device=self.config.device, requires_grad=False), 
                torch.tensor(data_items[10], device=self.config.device, requires_grad=False),
                torch.tensor(data_items[11], device=self.config.device, requires_grad=False),
                torch.tensor(data_items[-1], device=self.config.device, dtype=torch.long, requires_grad=False) if self.stage == 'train' else data_items[-1])

    def __iter__(self):
        self.indices = list(range(len(self.data)))
        if self.shuffle:
            random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.indices != []:
            idxs = self.indices[:self.batch_size]
            # log ids inside a batch
            batch = self._process(idxs)
            self.indices = self.indices[self.batch_size:]
            if self.config.enable_gpu == True:
                batch = self._cuda(batch)
            return batch
        else:
            raise StopIteration

    def get(self):
        self.reset()
        return self.__next__()

    def reset(self):
        self.indices = list(range(len(self.data)))
        if self.shuffle: random.shuffle(self.indices)
