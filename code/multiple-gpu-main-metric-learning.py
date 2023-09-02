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
import glob

from utilities import *
from prepare_data import get_UCRdataset, DatasetMetricLearning, BalancedBatchSampler
from model import ProposedModel
from loss import ContrastiveLoss
from eval import kNN

import lightning as L

log = logging.getLogger(__name__)


# @ hydra.main(config_path='conf', config_name='pre_training')
@ hydra.main(config_path='conf', config_name='metric_learning')

class unet(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ProposedModel(input_ch=dataset.channel).to(cfg.device)
        torchinfo.summary(
            self.model, (dataset.train_data[:1].shape, dataset.train_data[:1].shape), device=cfg.device)
        if cfg.pre_training:
            load_model_path = sorted(glob.glob(pre_trained_model_path+'*.pkl'))[-1]
            log.info('pre-trained model loading...')
            log.info('pre-trained model: '+load_model_path)
            model.load_state_dict(torch.load(
                self.load_model_path, map_location=cfg.device))

    def training_step(self, batch, batch_idx):
        data1, data2, sim = batch
        data1, data2 = data1.to(cfg.device), data2.to(cfg.device)
        sim = sim.to(cfg.device)
        y = self.model(data1, data2)
        loss, _ = loss_function(y, data1, data2, sim)
        train_losses.append(loss.item())
        return loss
    
    def validation_step(self, batch, batch_idx):
        val_ER, val_loss, _, _ = kNN(model, dataset, 'val', cfg)

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        train_loss = torch.mean(torch.FloatTensor(train_losses)).item()
        training_curve_loss.save(
            train_value=train_loss, val_value=val_loss)
        training_curve_ER.save(val_value=val_ER)
        save_model.save(model, val_ER)
        log.info('[%d/%d]-ptime: %.2f, train loss: %.4f, val loss: %.4f, val ER: %.4f'
             % ((epoch + 1), cfg.epoch, per_epoch_ptime, train_loss, val_loss, val_ER))
    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=cfg.lr, betas=(0.5, 0.999))