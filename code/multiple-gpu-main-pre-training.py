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

#log = logging.getLogger(__name__)

import omegaconf
import os

def Average(lst):
    return sum(lst) / len(lst)
class unet(L.LightningModule):
    def __init__(self,cfg):
        super().__init__()
        # define model & optimizer & loss function
        self.model = ProposedModel(input_ch=dataset.channel)
        self.model_summary = torchinfo.summary(
            self.model, (dataset.train_data[:1].shape, dataset.train_data[:1].shape), device=cfg.device, verbose=0)
        log.debug(self.model_summary)
        self.loss_function = nn.MSELoss()
        self.train_losses =[]
        self.val_losses=[]
        self.cfg=cfg

    def training_step(self, batch, batch_idx):
        data1, data2, path, _ =batch
        y = self.model(data1, data2)
        loss = self.loss_function(
            F.softmax(y, dim=2), F.softmax(path, dim=2))
        self.log("train_loss", loss,sync_dist=True)
        self.train_losses.append(loss.item())
        return loss
    
    def on_train_epoch_end(self):
        epoch_train_loss = Average(self.train_losses)
        self.log("epoch_train_loss",epoch_train_loss)
    
    def validation_step(self, batch, batch_idx):
        data1, data2, path, _ =batch
        y = self.model(data1, data2)
        loss = self.loss_function(
            F.softmax(y, dim=2), F.softmax(path, dim=2))
        self.val_losses.append(loss.item())
        self.log("validation_loss", loss,sync_dist=True)

    def on_validation_epoch_end(self):
        epoch_val_loss=Average(self.val_losses)
        print("")
        print("epoch_val_loss: ",epoch_val_loss)
        self.log("epoch_validation_loss",epoch_val_loss)
        self.val_losses.clear()

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.cfg.lr)

#c = []
#hydra.main(config_path='conf',config_name="pre_training")(lambda x:c.append(x))()
#cfg = c[0]
#print(cfg)


@ hydra.main(config_path='conf', config_name='pre_training')
def main(cfg: DictConfig) -> None:
    global dataset
    global cwd
    cwd = hydra.utils.get_original_cwd()+'/'
    fix_seed(cfg.seed)

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
    if dataset.N_train_data+dataset.N_val_data < cfg.dataset.used_dataset_threshold.num_train_data:
        exit()

    # If the length of data is more than the threshold, do not execute.
    if dataset.length > cfg.dataset.used_dataset_threshold.length_data:
        exit()

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

    # Lightning will automatically use all available GPUs!
    model = unet(cfg)
    trainer = L.Trainer(max_epochs=cfg.epoch, precision=16,gradient_clip_val=2.0)
    trainer.fit(model, train_dataloaders=train_loader,val_dataloaders = val_loader)



if __name__ == '__main__':
    main()