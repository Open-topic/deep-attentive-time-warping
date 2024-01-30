import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import torch
import torch.optim as optim
from utilities import *
from prepare_data import get_UCRdataset, DatasetMetricLearning, BalancedBatchSampler
from model import ProposedModel
from loss import ContrastiveLoss
from eval import kNN
import numpy as np

log = logging.getLogger(__name__)

@hydra.main(config_path='conf', config_name='metric_learning')
def main(cfg: DictConfig) -> None:
    fix_seed(cfg.seed)
    cwd = hydra.utils.get_original_cwd()+'/'

    # load data (split train data & standardization)
    dataset = get_UCRdataset(cwd, cfg)

    # make result folder
    result_path = '%s%sresult/' % (cwd, cfg.result_path)
    make_folder(path=result_path)
    result_path += '%s_%s/' % (str(cfg.dataset.ID).zfill(3), dataset.dataset_name)
    make_folder(path=result_path)
    pre_trained_model_path = result_path+'pre_training/'
    result_path += 'metric_learning/'
    make_folder(path=result_path)
    if not cfg.pre_training:
        result_path += 'wo_pre_training/'
        make_folder(result_path)
    result_path += '%s' % dataset.dataset_name

    # log saved at the result folder
    file_handler = logging.FileHandler('%s.log' % (result_path), 'a')
    log.addHandler(file_handler)

    log.info('\n=============================================================')
    log.debug(OmegaConf.to_yaml(cfg), cfg)
    log.info('dataset ID: %d, dataset name: %s' %
             (cfg.dataset.ID, dataset.dataset_name))

    # define model & optimizer & loss function
    model = ProposedModel(input_ch=dataset.channel).to(cfg.device)
    # if cfg.pre_training:
    #     load_model_path = sorted(glob.glob(pre_trained_model_path+'*.pkl'))[-1]
    #     log.info('pre-trained model loading...')
    #     log.info('pre-trained model: '+load_model_path)
    #     model.load_state_dict(torch.load(
    #         load_model_path, map_location=cfg.device))

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.5, 0.999))

    # test
    load_model_path = '/content/drive/MyDrive/deep-attentive-time-warping/result/001_Adiac/metric_learning/Adiac_2024.01.15.12.48.44_lr_0.0001_ProposedModel_epoch_13_ER_0.3421.pkl'
    log.info('test model loading...')
    log.info('test model: '+load_model_path)
    model.load_state_dict(torch.load(
        load_model_path, map_location=cfg.device))
    test_ER, test_loss, pred, neighbor, distall = kNN(model, dataset, 'test', cfg)
    log.info('test loss: %.4f, test ER: %.4f' % (test_loss, test_ER))
    print(distall)

    log.info(pred)
    log.info(neighbor)
  

    # Save neighbor array to a text file
    dist_file_path = '/content/drive/MyDrive/deep-attentive-time-warping/code/distance_all.txt'
    np.savetxt(dist_file_path, distall, fmt='%d')
    log.info('Distance_all array saved to %s' % dist_file_path)

if __name__ == '__main__':
    main()
