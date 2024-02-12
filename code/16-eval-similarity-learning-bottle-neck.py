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
from model.proposed_unet_bottle_neck import ProposedModel
from loss import ContrastiveLoss
from eval import kNN_Similarity_Only
#from eval import kNNMixed


log = logging.getLogger(__name__)
use_amp = True
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
#device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

from accelerate import Accelerator
accelerator = Accelerator(mixed_precision="fp16")
device_type = accelerator.device


# @ hydra.main(config_path='conf', config_name='pre_training')
@ hydra.main(config_path='conf', config_name='metric_learning')
def main(cfg: DictConfig) -> None:
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
    pre_trained_model_path = result_path+'metric_learning/'
    result_path += 'similarity_learning/'
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

    # define model & optimizer & loss function
    model = ProposedModel(input_ch=dataset.channel)

    try:
        torchinfo.summary(model, (dataset.train_data[:1].shape, dataset.train_data[:1].shape), device=cfg.device)
    except:
        print('cannot show model summary')

    torchinfo.summary(
        model, (dataset.train_data[:1].shape, dataset.train_data[:1].shape), device=cfg.device)
    if cfg.pre_training:
        load_model_path = sorted(glob.glob(pre_trained_model_path+'*.pkl'))[-1]
        log.info('metric-trained model loading...')
        log.info('metric-trained model: '+load_model_path)
        model.load_state_dict(torch.load(
            load_model_path, map_location=cfg.device))

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
    loss_function = ContrastiveLoss(cfg.tau)

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
    log.info('Batch size: {}'.format(cfg.batch_size))

    model = model.to(device_type)
    # Freeze and require no grade on everything except the projector
    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze weight and bias of the projector
    for param in model.unet.projector.parameters():
        param.requires_grad = True

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    # test
    load_model_path = sorted(glob.glob(result_path+'*.pkl'))[-1]
    log.info('test model loading...')
    log.info('test model: '+load_model_path)
    model.load_state_dict(torch.load(
        load_model_path, map_location=cfg.device))
    model.eval()
    # test_ER, test_loss, pred, neighbor = kNNMixed.kNNMixed(model, dataset, 'test', cfg)
    test_ER, test_loss, pred, neighbor = kNN_Similarity_Only.kNN_Similarity_Only(model, dataset, 'test', cfg)
    log.info('test loss: %.4f, test ER: %.4f' % (test_loss, test_ER))

if __name__ == '__main__':
    main()