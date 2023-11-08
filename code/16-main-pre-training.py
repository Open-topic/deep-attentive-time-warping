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

from accelerate import Accelerator
accelerator = Accelerator(mixed_precision="fp16")
device_type = accelerator.device


log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)

use_amp = True
# scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
#device_type = accelerator.device
# torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(torch_device)

@ hydra.main(config_path='conf', config_name='pre_training')
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
    model = ProposedModel(input_ch=dataset.channel)
    try:
        model_summary = torchinfo.summary(
            model, (dataset.train_data[:1].shape, dataset.train_data[:1].shape), device=cfg.device, verbose=0)
        log.debug(model_summary)

    except:
        print('cannot show model summary')

    # # Determine_batch_size by power method
    # for i in range(100):
    #           try:
    #             print("power_batch_size:",2**i)
    #             data_in_shape = dataset.train_data[:1].shape
    #             data_in_shape = list(data_in_shape)
    #             data_in_shape[0] = 2**i
    #             print("data_in_shape:",data_in_shape)
    #             data_in_shape = tuple(data_in_shape)
    #             model_summary = torchinfo.summary(
    #                 model, (data_in_shape, data_in_shape), device=cfg.device, verbose=0)
    #           except:
    #             # log.debug(model_summary)
    #             log.debug("power_batch_size", 2**i)
    #             break
    # Power method fail, we need to try forward/backward

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
    loss_function = nn.MSELoss()

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
    log.info('Batch size: {}'.format(cfg.batch_size))

    # train
    date = get_date()
    log.info('data: '+date)
    save_name = '_%s_lr_%s' % (date, cfg.lr)
    log.info('save_name: '+save_name)
    training_curve_loss = TrainingCurve(
        'loss', result_path+save_name, cfg)
    save_model = SaveModel('loss', 'less', result_path+save_name, cfg)

    model = model.to(device_type)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    val_loader = accelerator.prepare_data_loader(val_loader)

    epoch = 0
    fix_seed(cfg.seed)
    while epoch < cfg.epoch:
        # train
        model.train()
        train_losses = []
        epoch_start_time = time.time()
        for data1, data2, path, _ in tqdm(train_loader):
            optimizer.zero_grad()
            with accelerator.autocast():
                data1, data2 = data1, data2
                path = path
                y = model(data1, data2)
                loss = loss_function(F.softmax(y, dim=2), F.softmax(path, dim=2))
            accelerator.backward(loss)
            optimizer.step()
            train_losses.append(loss.item())

        # val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for data1, data2, path, _ in tqdm(val_loader):
                with accelerator.autocast():
                    path = path
                    y = model(data1, data2)
                    loss = loss_function(
                        F.softmax(y, dim=2), F.softmax(path, dim=2))
                val_losses.append(loss.item())

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
        train_loss = torch.mean(torch.FloatTensor(train_losses)).item()
        val_loss = torch.mean(torch.FloatTensor(val_losses)).item()

        training_curve_loss.save(train_value=train_loss, val_value=val_loss)
        save_model.save(model, val_loss)
        log.info('[%d/%d]-ptime: %.2f, train loss: %.4f, val loss: %.4f'
                 % ((epoch + 1), cfg.epoch, per_epoch_ptime, train_loss, val_loss))
        epoch += 1


if __name__ == '__main__':
    main()
