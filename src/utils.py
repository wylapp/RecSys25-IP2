from sklearn.metrics import roc_auc_score
import numpy as np
import re
import argparse
import torch
import os

def cal_steps(data_len, config, stage="pretrain"):
    if stage == "pretrain":
        if data_len % config.pretrain_batch == 0:
            return (data_len // int(config.pretrain_batch)) * int(config.pt_epoch)
        else:
            return (data_len // int(config.pretrain_batch) + 1) * int(config.pt_epoch)
    else:
        if data_len % config.batch_size == 0:
            # TODO change it before release. Present working fine settings.
            return (data_len // int(config.batch_size)) * int(config.epochs)
        else:
            return (data_len // int(config.batch_size) + 1) * int(config.epochs)


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def dcg_score(y_true, y_score, k=10) -> float:
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10) -> float:
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score) -> float:
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def ctr_score(y_true, y_score, k=1):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    return np.mean(y_true)

def acc(y_true, y_hat):
    y_hat = torch.argmax(y_hat, dim=-1)
    tot = y_true.shape[0]
    hit = torch.sum(y_true == y_hat)
    return hit.data.float() * 1.0 / tot



def word_tokenize(sent: str) -> list:
    """Split sentence into tokens, not only rely on blank spaces,
    punctuation marks included.
    """
    pat = re.compile(r'[\w]+|[.,!?;|]')
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []

def latest_checkpoint(directory, stage="pre"):
    assert stage in ["pre", "downstream"], "invalid task stage"
    if not os.path.exists(directory):
        raise FileNotFoundError
    
    alt_files = [fi for fi in os.listdir(directory) if fi.startswith(stage)]
    if len(alt_files)==0:
        raise FileNotFoundError
    
    return os.path.join(directory, sorted(alt_files)[-1])

def get_checkpoint(directory, ckpt_name):
    ckpt_path = os.path.join(directory, ckpt_name)
    if os.path.exists(ckpt_path):
        return ckpt_path
    else:
        raise FileNotFoundError
    
def save_checkpoit(model, directory, stage, epoch):
    torch.save(model.state_dict(), os.path.join(directory, f'{stage}-{epoch}.pth'))


def calculate_single_user_metric(pair):
    try:
        auc = roc_auc_score(*pair)
        mrr = mrr_score(*pair)
        ndcg5 = ndcg_score(*pair, 5)
        ndcg10 = ndcg_score(*pair, 10)
        return [auc, mrr, ndcg5, ndcg10]
    except ValueError:
        return [np.nan] * 4
    
def weight_norm(alpha: int, beta: int, delta:int):
    """Normalize weight for contrasitive learning losses
    """
    wsum = alpha+beta+delta
    return alpha/wsum, beta/wsum, delta/wsum
