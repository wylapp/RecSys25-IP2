from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from loguru import logger
import random
import pickle

def preprocess_behav(config, stage:str) -> bool:
    data_dir = 'data'
    dataset = config.dataset
    behav_path = os.path.join(data_dir, dataset, stage, 'behaviors.tsv')
    
    lis_files = os.listdir(os.path.join(data_dir, dataset, stage))
    if 'behavior_parsed.tsv' in lis_files and 'user2int.tsv' in lis_files and 'news2id.pkl' in lis_files and config.force_new != True:
        logger.warning("Preprocessed files found! Exit prep stage. Don't use outdated files!")
        return True
    
    logger.info(f"Process behavior data: {behav_path}")
    
    # * * * newsID map to index * * *
    news = pd.read_table(os.path.join("data", config.dataset, stage, 'news.tsv'), sep='\t', header=None, names=[
                  'id', 'category', 'subcategory', 'title', 'abstract', 'url',
                  'title_entities', 'abstract_entities'
              ], quotechar=" ")
    news_id_map = dict(zip(news.id.tolist(), range(len(news))))
    with open(os.path.join(data_dir, dataset, stage, "news2id.pkl"), "wb") as pf:
        pickle.dump(news_id_map, pf)

    # * * * Behavior file parse * * *
    behaviors = pd.read_table(
        behav_path,
        header=None,
        names=['impression_id', 'user', 'time', 'clicked_news', 'impressions'])
    behaviors.impressions = behaviors.impressions.str.split()

    user2int = {}
    for row in behaviors.itertuples(index=False):
        if row.user not in user2int:
            user2int[row.user] = len(user2int) + 1
    
    user2int_path = os.path.join(data_dir, dataset, stage, "user2int.tsv")
    pd.DataFrame(user2int.items(), columns=['user',
                                            'int']).to_csv(user2int_path,
                                                           sep='\t',
                                                           index=False)
    behaviors.dropna(subset=['clicked_news'], inplace=True)
    behaviors.dropna(subset=['impressions'], inplace=True)
    behaviors = behaviors.reset_index(drop=True)
    # change userID into index IDs
    for row in behaviors.itertuples():
        behaviors.at[row.Index, 'user'] = user2int[row.user]
        sep_hist = row.clicked_news.split(' ')
        if len(sep_hist) > config.user_log_length:
            sep_hist = sep_hist[-config.user_log_length:]
            behaviors.at[row.Index, 'clicked_news'] = " ".join(sep_hist)
    # negative sampling
    if stage == 'train':
        for row in tqdm(behaviors.itertuples(), desc="Balancing data"):
            positive = iter([x for x in row.impressions if x.endswith('1')])
            negative = [x for x in row.impressions if x.endswith('0')]
            random.shuffle(negative)
            negative = iter(negative)
            pairs = []
            try:
                while True:
                    pair = [next(positive)]
                    for _ in range(config.npratio):
                        pair.append(next(negative))
                    pairs.append(pair)
            except StopIteration:
                pass
            behaviors.at[row.Index, 'impressions'] = pairs
        # unfold rows with multiple positive samples
        behaviors = behaviors.explode('impressions').dropna(
            subset=["impressions"]).reset_index(drop=True)
    behaviors[['candidate_news', 'clicked']] = pd.DataFrame(
        behaviors.impressions.map(
            lambda x: (' '.join([e.split('-')[0] for e in x]), ' '.join(
                [e.split('-')[1] for e in x]))).tolist())
    
    target = behav_path = os.path.join(data_dir, dataset, stage, 'behavior_parsed.tsv')
    behaviors.to_csv(
        target,
        sep='\t',
        index=False,
        columns=['user', 'clicked_news', 'candidate_news', 'clicked'])
    logger.success(f"Process behavior data done!")
    return True


def category_mapping(config) -> bool:
    if config.force_new != True:
        return True

    news = pd.read_table(os.path.join(config.data_dir, config.dataset, "train", 'news.tsv'), sep='\t', header=None, names=[
                  'id', 'category', 'subcategory', 'title', 'abstract', 'url',
                  'title_entities', 'abstract_entities'
              ], quotechar=" ")
    test_news = pd.read_table(os.path.join(config.data_dir, config.dataset, "test", 'news.tsv'), sep='\t', header=None, names=[
                  'id', 'category', 'subcategory', 'title', 'abstract', 'url',
                  'title_entities', 'abstract_entities'
              ], quotechar=" ")
    
    join_cat = set(news.category.unique()) | set(test_news.category.unique())
    join_subcat = set(news.subcategory.unique()) | set(test_news.subcategory.unique())

    lt_cat = {cat: i for i, cat in enumerate(join_cat)}
    lt_subcat = {cat: i for i, cat in enumerate(join_subcat)}
    
    for stage in ["train", "test"]:
        with open(os.path.join(config.data_dir, config.dataset, stage, "cat2id.pkl"), "wb") as pf:
            pickle.dump(lt_cat, pf)

        with open(os.path.join(config.data_dir, config.dataset, stage, "subcat2id.pkl"), "wb") as pf:
            pickle.dump(lt_subcat, pf)

    return True

def entity_embedding_mapping(config) -> int:
    if config.force_new != True:
        return True
    embedding_grid = pd.read_table(os.path.join(config.data_dir, config.dataset, "train", "entity_embedding.vec"), sep='\t', header=None, )
    embedding_grid_test = pd.read_table(os.path.join(config.data_dir, config.dataset, "test", "entity_embedding.vec"), sep='\t', header=None, )
    
    merged = pd.concat([embedding_grid, embedding_grid_test], axis=0, ignore_index=True)
    merged.drop_duplicates(subset=0, inplace=True)

    padding_pos = np.zeros((1, config.ent_dim))
    entities_emb = merged.iloc[:, 1:-1].values.astype(np.float32)
    
    # * build the entiy2id cast table. Counting from 1, position 0 reserved for padding
    entity_vocab = merged[0].tolist()
    vocab_dict = dict(zip(entity_vocab, range(1, len(entity_vocab)+1)))

    news = pd.read_table(os.path.join(config.data_dir, config.dataset, "train", 'news.tsv'), sep='\t', header=None, names=[
                  'id', 'category', 'subcategory', 'title', 'abstract', 'url',
                  'title_entities', 'abstract_entities'
              ], quotechar=" ")
    test_news = pd.read_table(os.path.join(config.data_dir, config.dataset, "test", 'news.tsv'), sep='\t', header=None, names=[
                  'id', 'category', 'subcategory', 'title', 'abstract', 'url',
                  'title_entities', 'abstract_entities'
              ], quotechar=" ")
    
    join_cat = set(news.category.unique()) | set(test_news.category.unique())
    join_subcat = set(news.subcategory.unique()) | set(test_news.subcategory.unique())

    # * build the cat2id cast table. Counting start from #entities
    lt_cat = dict(zip(join_cat, range(len(entity_vocab), len(entity_vocab)+len(join_cat))))
    lt_subcat = dict(zip(join_subcat, range(len(entity_vocab)+len(join_cat), len(entity_vocab)+len(join_cat)+len(join_subcat))))

    # * write all cast tables to multiple positions
    for stage in ["train", "test"]:
        with open(os.path.join(config.data_dir, config.dataset, stage, "cat2id.pkl"), "wb") as pf:
            pickle.dump(lt_cat, pf)

        with open(os.path.join(config.data_dir, config.dataset, stage, "subcat2id.pkl"), "wb") as pf:
            pickle.dump(lt_subcat, pf)

        with open(os.path.join(config.data_dir, config.dataset, stage, "ent2id.pkl"), "wb") as pf:
            pickle.dump(vocab_dict, pf)

    catgory_embs = np.random.rand(len(lt_cat), config.ent_dim)
    subcat_embs = np.random.rand(len(lt_subcat), config.ent_dim)

    full_embeddings = np.concatenate([padding_pos, entities_emb, catgory_embs, subcat_embs])

    # * only write to one place, since embedding initialization only happen once.
    full_embeddings.dump(os.path.join(config.data_dir, config.dataset, stage, "entemb.pkl"))

    return None