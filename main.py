import argparse
import yaml
from attrdict import AttrDict
import os
import time
import json
import torch
from src import common
from loguru import logger
from src import utils
from src.dataset import *
from src.preprocess import preprocess_behav, entity_embedding_mapping
from src.main_model import *
from tqdm import tqdm
from src.utils import calculate_single_user_metric, weight_norm, str2bool
from multiprocessing import Pool
from torch.utils.tensorboard import SummaryWriter
from src.optimizer import *
from src.loss import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_config(config_file):
    with open(config_file, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as e:
            print(f"Error while loading config file: {e}")


class Mainloop():

    def __init__(self, args) -> None:
        self.config = args
        config = AttrDict(
            yaml.load(open('config.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader))

        for k, v in vars(args).items():
            setattr(config, k, v)

        common.set_seed(config.seed)
        self.config = config
        # self.config is the only global configurator that holds all settings.
        if 'dry-run' in self.config.mode and os.name == 'posix':
            self.writer = SummaryWriter("/tmp")
        else:
            logger.add("log/log_{time}.log", level='DEBUG', rotation='5 MB')
            self.writer = SummaryWriter()
        logger.debug("Log start")
        logger.info(self.config)
        logger.warning(f"Regenerate files? {config.force_new}")
        self.writer.add_text('train/setups', f"```{str(config)}```")

        # set the device into pytorch attribute, because hparams writer only accepts str
        self.config.device = torch.device('cuda:{}'.format(
            config.cuda_index) if torch.cuda.is_available() else 'cpu')
        # if 'train' in self.config.mode:
        self._init_tester()
        self.scaler = torch.cuda.amp.GradScaler()

    def _init_tester(self, mode="test"):
        preprocess_behav(self.config, mode)
        entity_embedding_mapping(self.config)
        news_dataset = newsArchiveDataset(self.config, mode)
        downstream_ds = MindDataset(self.config, mode, news_dataset)
        logger.success("Downstream test dataset ready!")
        self.config.emb_size = news_dataset.emb_size
        self.tester_dl = MINDloader(downstream_ds, self.config, shuffle=True, stage=mode)

    def pretrain_iter(self):
        """
        Pretrain stage on the training news set. Due to the nature of Self-supervised Learning (SSL),
        there is no validation or test stages.
        """
        pt_dataset = newsArchiveDataset(self.config)
        logger.info("data loaded")
        pt_loader = preDataLoader(pt_dataset, self.config)
        # *prepare data
        
        model = Ent_Pretrainer(self.config).to(self.config.device)
        num_training_steps = utils.cal_steps(len(pt_dataset), self.config)
        optimizer, scheduler = build_optimizer(model, num_training_steps, self.config, stage="pretrain")
        logger.info("Pretrain start")
        loss_func = NT_xent(self.config.temp)
        loss_title = NoisyLoss(self.config.temp)
        loss_ent = NoisyLoss(self.config.temp)
        model.train()
        model._freeze_partial_bert()

        a, b, c = weight_norm(self.config.alpha, self.config.beta, self.config.delta)
        best_loss = float('inf')
        for ep in range(self.config.pt_epoch):
            loss = 0
            pbar = tqdm(enumerate(pt_loader))
            for i, ite in pbar:
                if ite[2].shape[0] == 0:
                    logger.error("All negative happens!")
                    # no valid entity embeddings in this batch.
                    continue
                local_data = model(ite)
                local_loss = loss_func(local_data[:3])
                local_ent_loss = loss_ent((local_data[1], local_data[4]))
                local_title_loss = loss_title((local_data[0], local_data[3]))
                # * intermedia results preserved in the model,
                # then call contrastive learning
                local_closs =  a*local_loss + b*local_title_loss + c*local_ent_loss
                # local_closs = a*local_loss
                loss += local_closs.item()
                local_closs.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if i%10 == 0:
                    pbar.set_description(f'Current loss (in step): {local_closs:.5f}')
            logger.warning(f'Ep.{ep},loss:{loss:6f}')
            if loss < best_loss:
                best_loss = loss
                utils.save_checkpoit(model, self.config.model_dir, "pre", ep)

    
    def train_iter(self):
        # * prepare data
        preprocess_behav(self.config, "train")
        news_dataset = newsArchiveDataset(self.config)
        downstream_ds = MindDataset(self.config, "train", news_dataset)
        logger.success("Downstream task dataset ready!")
        downstream_dl = MINDloader(downstream_ds, self.config, shuffle=True)
        # * init model
        model = IP2(self.config).to(self.config.device)
        
        # * load pretrained news encoder
        if self.config.pretrain == True:
            pt_ckp_path = utils.latest_checkpoint(self.config.model_dir, stage="pre")
            pt_statedict = torch.load(pt_ckp_path, self.config.device)
            model._load_pretrain_encoder(pt_statedict)
        model._freeze_partial_bert()

        num_training_steps = utils.cal_steps(len(downstream_ds), self.config, stage="train")
        optimizer, scheduler = build_optimizer(model, num_training_steps, self.config)
        loss_func = nn.CrossEntropyLoss()
        # ! train starts here
        best_loss = float('inf')
        model.train()
        logger.debug("Downstream train start!")
        for ep in range(self.config.epochs):
            loss = 0
            pbar = tqdm(enumerate(downstream_dl))
            for i, ds in pbar:
                optimizer.zero_grad()

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    local_data = model(ds)
                    local_loss = loss_func(local_data, ds[-1])
                # * intermedia results preserved in the model,
                # then call contrastive learning
                loss += local_loss.item()
                # local_loss.backward()
                # optimizer.step()
                self.scaler.scale(local_loss).backward()
                self.scaler.step(optimizer=optimizer)
                self.scaler.update()

                scheduler.step()
            
                if i%10 == 0:
                    pbar.set_description(f'Current loss (in step): {local_loss:.5f}')
            logger.warning(f'Ep.{ep}, loss:{loss:.6f}')
            aucs, mrrs, ndcg5s, ndcg10s = self.test_iter(model=model)
            self.writer.add_scalar('train/loss', loss, ep)
            self.writer.add_scalar('test/auc', aucs, ep)
            self.writer.add_scalar('test/mrr', mrrs, ep)
            self.writer.add_scalar('test/ndcg5', ndcg5s, ep)
            self.writer.add_scalar('test/ndcg10', ndcg10s, ep)
            self.writer.add_scalar('train/lr', scheduler.get_last_lr()[0], ep)

            if loss < best_loss:
                best_loss = loss
                # utils.save_checkpoit(model, self.config.model_dir, ep)

    def test_iter(self, model=None, eval=True):
        model.eval()
        pbar = tqdm(enumerate(self.tester_dl))
        pairs = []
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                for i, ds in pbar:
                    hist_vec, hist_mask, ent_vec = model.get_news_vector((ds[0], ds[2], ds[6], ds[10]), ds[4])
                    imp_vec, imp_ent = model.get_news_vector_reduced((ds[1], ds[3], ds[7], ds[11]), ds[5])
                    hist_user_vec, hist_user_vec2= model.get_user_vector(hist_vec, hist_mask, ent_vec)

                    pred_digit = model.get_prediction_reduced(imp_vec, imp_ent, ds[5], hist_user_vec, hist_user_vec2)
                    pred_logits = []
                    for row, interval in zip(pred_digit, ds[5]):
                        pred_logits.append(row.tolist())
                    pairs.extend(list(zip(ds[-1], pred_logits)))

        with Pool(processes=self.config.num_workers) as pool:
            results = pool.map(calculate_single_user_metric, pairs)
        aucs, mrrs, ndcg5s, ndcg10s = np.array(results).T
        
        logger.warning(f"AUC: {np.nanmean(aucs):.5f} MRR: {np.nanmean(mrrs):.5f} nDCG@5: {np.nanmean(ndcg5s):.5f} nDCG@10: {np.nanmean(ndcg10s):.5f}")
        if eval is True:
            model.train()
        
        return np.nanmean(aucs), np.nanmean(mrrs), np.nanmean(ndcg5s), np.nanmean(ndcg10s)

    def dry_run(self):
        r"""
        Dry-run for new feature test!
        """
        preprocess_behav(self.config, "train")
        news_dataset = newsArchiveDataset(self.config)
        downstream_ds = MindDataset(self.config, "train", news_dataset)
        logger.success("Downstream task dataset ready!")
        downstream_dl = MINDloader(downstream_ds, self.config, shuffle=True)
        for i, ds in enumerate(downstream_dl):
            print(ds)
            time.sleep(3)

    def exec(self):
        # ! Main entrance
        if str2bool(self.config.pretrain) == True:
            logger.warning("Pre-train stage")
            self.pretrain_iter()
        if 'train' in self.config.mode:
            self.train_iter()
        if 'test' in self.config.mode:
            model = IP2(self.config).to(self.config.device)
            model._freeze_partial_bert()
            self.test_iter(model=model)
        if 'dry-run' in self.config.mode:
            logger.warning("Dry-run for new feature test!")
            self.dry_run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train_test",
                        choices=['train', 'test', 'train_test', 'dry-run'])
    parser.add_argument("--pretrain", type=str2bool, default=True)
    parser.add_argument("--root_data_dir", type=str, default="data")
    parser.add_argument("--pooler_type",
                        type=str,
                        default='attention',
                        choices=['avg', 'attention', 'cls', 'avg_first_last', 'avg_top2', 'gate'])
    parser.add_argument("--predict_type", type=str, default='cross',
                        choices=['avg', 'cross', 'max', 'cast', 'text', 'entity'])
    args = parser.parse_args()
    mainloop = Mainloop(args)
    mainloop.exec()
