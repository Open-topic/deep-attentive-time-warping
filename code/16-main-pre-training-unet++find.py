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
from model.proposed_unet_2Plus import ProposedModel

from accelerate import Accelerator
accelerator = Accelerator(mixed_precision="fp16")
device_type = accelerator.device


log = logging.getLogger(__name__)
use_amp = True
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
#device_type = accelerator.device

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
        print(model_summary)
    except:
        print('cannot show model summary')
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

    # Find the correct batch_size via power method
    for power in range(20):
        data1, data2, path, sim = train_dataset.__getitem__(0)
        data_shape = list(data1.shape)
        path_shape = list(path.shape)
        power_batch_size = 2**power
        data_shape.insert(0,power_batch_size)
        path_shape.insert(0,power_batch_size)
        print("data_shape",data_shape)
        print("path_shape",path_shape)
        find_batch_size = 2**power
        cfg.batch_size = find_batch_size
        dataset = get_UCRdataset(cwd, cfg)
        train_dataset = DatasetPreTraining(dataset, 'train', cfg)  
        try:
            model.train()
            train_losses = []
            epoch_start_time = time.time()
            print("tried power =", power)
            for _ in range(5):
                optimizer.zero_grad()
                data1 = torch.rand(data_shape).to(accelerator.device)
                data2 = torch.rand(data_shape).to(accelerator.device)
                path = torch.rand(path_shape).to(accelerator.device)
                print("check point")
                with accelerator.autocast():
                    y = model(data1, data2)
                    loss = loss_function(F.softmax(y, dim=2), F.softmax(path, dim=2))
                accelerator.backward(loss)
                optimizer.step()
                train_losses.append(loss.item())
                
                break
        except Exception as error:
            print(error)
            print("power: ", power-1)
            break

if __name__ == '__main__':
    main()