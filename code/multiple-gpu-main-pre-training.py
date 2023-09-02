import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchinfo
from tqdm import tqdm
import time

from utilities import *
from prepare_data import get_UCRdataset, DatasetPreTraining, BalancedBatchSampler
from model import ProposedModel

import lightning as L

log = logging.getLogger(__name__)


@ hydra.main(config_path='conf', config_name='pre_training')
class unet(L.LightningModule):
    def __init__(self):
        super().__init__()
        fix_seed(cfg.seed)
        cwd = hydra.utils.get_original_cwd()+'/'

        # load data (split train data & standardizarion)
        dataset = get_UCRdataset(cwd, cfg)

        # make result folder
        result_path = '%s%sresult/' % (cwd, cfg.result_path)
        make_folder(path=result_path)
        result_path += '%s_%s/' % (str(cfg.dataset.ID).zfill(3),
                                dataset.dataset_name)
        make_folder(path=result_path)
        result_path += 'pre_training/'
        make_folder(path=result_path)
        result_path += '%s' % dataset.dataset_name

        # log saved at result folder
        file_handler = logging.FileHandler(
            '%s.log' % (result_path), 'a')
        log.addHandler(file_handler)

        log.info('\n=============================================================')
        log.debug(OmegaConf.to_yaml(cfg), cfg)
        log.info('dataset ID: %d, dataset name: %s' %
                (cfg.dataset.ID, dataset.dataset_name))

        # If the number of training + validation data is less than the threshold, do not execute.
        log.info('Number of training + validation data: %d' %
                (dataset.N_train_data+dataset.N_val_data))
        if dataset.N_train_data+dataset.N_val_data < cfg.dataset.used_dataset_threshold.num_train_data:
            log.info('The number of training data is less than %d ...' %
                    cfg.dataset.used_dataset_threshold.num_train_data)
            log.info('It is not executed.')
            exit()

        # If the length of data is more than the threshold, do not execute.
        log.info('Length of data: %d' % dataset.length)
        if dataset.length > cfg.dataset.used_dataset_threshold.length_data:
            log.info('The number of training data is more than %d ...' %
                    cfg.dataset.used_dataset_threshold.length_data)
            log.info('It is not executed.')
            exit()

        # define model & optimizer & loss function
        model = ProposedModel(input_ch=dataset.channel).to(cfg.device)
        model_summary = torchinfo.summary(
            model, (dataset.train_data[:1].shape, dataset.train_data[:1].shape), device=cfg.device, verbose=0)
        log.debug(model_summary)
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
        self.loss_function = nn.MSELoss()
        self.train_losses =[]
        self.val_losses=[]

    def training_step(self, batch, batch_idx):
        data1, data2, path, _ =batch
        data1, data2 = data1.to(cfg.device), data2.to(cfg.device)
        path = path.to(cfg.device)
        y = model(data1, data2)
        loss = self.loss_function(
            F.softmax(y, dim=2), F.softmax(path, dim=2))
        self.log("train_loss", loss)
        training_losses.append(loss.item())
        return loss
    
    def validation_step(self, batch, batch_idx):
        data1, data2, path, _ =batch
        data1, data2 = data1.to(cfg.device), data2.to(cfg.device)
        path = path.to(cfg.device)
        y = model(data1, data2)
        loss = loss_function(
            F.softmax(y, dim=2), F.softmax(path, dim=2))
        val_losses.append(loss.item())
        self.log("validation_loss", loss)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=cfg.lr, betas=(0.5, 0.999))

#c = []
#hydra.main(config_path='conf',config_name="pre_training")(lambda x:c.append(x))()
#cfg = c[0]
#print(cfg)

cwd = hydra.utils.get_original_cwd()+'/'
dataset = get_UCRdataset(cwd, cfg)

# make data loader
# train
train_dataset = DatasetPreTraining(dataset, 'train', cfg)
if cfg.train_loader_balance:
    train_batch_sampler = BalancedBatchSampler(
        dataset, 'train', cfg.positive_ration, cfg.negative_ration, cfg)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=train_batch_sampler, num_workers=cfg.num_workers)
else:
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True)
log.info('Length of train_loader: %d' % len(train_loader))

# val
val_dataset = DatasetPreTraining(dataset, 'val', cfg)
if cfg.val_loader_balance:
    val_batch_sampler = BalancedBatchSampler(
        dataset, 'val', cfg.positive_ration, cfg.negative_ration, cfg)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_sampler=val_batch_sampler, num_workers=cfg.num_workers)
else:
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True)
log.info('Length of val_loader: %d' % len(val_loader))

# Lightning will automatically use all available GPUs!
trainer = L.Trainer()
trainer.fit(unet(), train_dataloader=train_loader,val_dataloader = val_loader)