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

def Average(lst):
    return sum(lst) / len(lst)


# @ hydra.main(config_path='conf', config_name='pre_training')
@ hydra.main(config_path='conf', config_name='metric_learning')

class unet(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ProposedModel(input_ch=dataset.channel).to(cfg.device)
        if cfg.pre_training:
            load_model_path = sorted(glob.glob(pre_trained_model_path+'*.pkl'))[-1]
            log.info('pre-trained model loading...')
            log.info('pre-trained model: '+load_model_path)
            model.load_state_dict(torch.load(
                self.load_model_path, map_location=cfg.device))
        
        self.train_losses = []

    def training_step(self, batch, batch_idx):
        data1, data2, sim = batch
        y = self.model(data1, data2)
        loss, _ = loss_function(y, data1, data2, sim)
        self.train_losses.append(loss.item())
        return loss
        
    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
    
    def on_train_epoch_end(self):

        val_ER, val_loss, _, _ = kNN(self.model, dataset, 'val', cfg)

        epoch_end_time = time.time()
        # per_epoch_ptime = epoch_end_time - epoch_start_time

        train_loss = torch.mean(torch.FloatTensor(self.train_losses)).item()
        training_curve_loss.save(
            train_value=train_loss, val_value=val_loss)
        training_curve_ER.save(val_value=val_ER)
        save_model.save(model, val_ER)
        # log.info('[%d/%d]-ptime: %.2f, train loss: %.4f, val loss: %.4f, val ER: %.4f'
        #      % ((epoch + 1), cfg.epoch, per_epoch_ptime, train_loss, val_loss, val_ER))
        self.log("train_loss", train_loss)
        self.log("val_loss", val_loss)
        self.log("val_ER", val_ER)

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

    pre_trained_model_path = result_path+'pre_training/'
    result_path += 'metric_learning/'

    pre_trained_model_path = result_path+'pre_training/'
    result_path += 'metric_learning/'
    make_folder(path=result_path)
    if not cfg.pre_training:
        result_path += 'wo_pre_training/'
        make_folder(result_path)
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

    # make data loader
    # train
    train_dataset = DatasetMetricLearning(dataset)
    if cfg.train_loader_balance:
        train_batch_sampler = BalancedBatchSampler(
            dataset, 'train', cfg.positive_ration, cfg.negative_ration, cfg)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_sampler=train_batch_sampler, num_workers=cfg.num_workers)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True)
    log.info('Length of train_loader: %d' % len(train_loader))

    # Lightning will automatically use all available GPUs!
    trainer = L.Trainer(max_epochs=cfg.epoch, precision=16,gradient_clip_val=2.0)
    trainer.fit(unet(cfg), train_dataloaders=train_loader)

    # test
    load_model_path = sorted(glob.glob(result_path+'*.ckpt'))[-1]
    log.info('test model loading...')
    log.info('test model: '+load_model_path)
    # model.load_state_dict(torch.load(
    #     load_model_path, map_location=cfg.device))
    unet.load_from_checkpoint(load_model_path)
    test_ER, test_loss, pred, neighbor = kNN(model, dataset, 'test', cfg)
    log.info('test loss: %.4f, test ER: %.4f' % (test_loss, test_ER))

if __name__ == '__main__':
    main()