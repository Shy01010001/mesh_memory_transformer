import os
import argparse

import numpy as np
import torch

from models.models import MMTModel
from modules.dataloaders import R2DataLoader
from modules.loss import compute_loss
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.tokenizers import Tokenizer
from modules.trainer import Trainer
# from modules.base_cmn import get_dict
# import json


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='data/iu_xray/images/',
                        help='the path to the directory containing the data')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation.json',
                        help='the path to the directory containing the data')

    parser.add_argument('--folds', type=str, help='', default="")
    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'],
                        help='the dataset to be used')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True,
                        help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer')
    parser.add_argument('--d_ff', type=int, default=2048, help='the dimension of FFN')
    parser.add_argument('--d_vf', type=int, default=1024, help='the dimension of the patch features')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer')
    parser.add_argument('--num_layers', type=int, default=2, help='the number of layers of Transformer')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer')

    # !!! for semantic mapping
    parser.add_argument('--num_classes', type=int, default=20, help='the number disease classes')
    parser.add_argument('--num_organs', type=int, default=9, help='the number organs')
    parser.add_argument('--num_diseases_per_organ', type=int, default=8, help='the number diseases per organ')

    # for Cross-modal Memory
    parser.add_argument('--topk', type=int, default=1, help='the number of k')
    parser.add_argument('--cmm_size', type=int, default=2048, help='cmm size')
    parser.add_argument('--cmm_dim', type=int, default=512, help='cmm dimension')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image')
    parser.add_argument('--group_size', type=int, default=1, help='the group size')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used')
    parser.add_argument('--epochs', type=int, default=200, help='the number of training epochs')
    parser.add_argument('--save_dir', type=str, default='results/iu_xray', help='the patch to save the models')
    parser.add_argument('--record_dir', type=str, default='records/', help='the patch to save the results of experiments')
    parser.add_argument('--log_period', type=int, default=1000, help='the logging interval (in batches)')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period (in epochs)')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'],
                        help='whether to max or min the metric')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored')
    parser.add_argument('--early_stop', type=int, default=100, help='the patience of training')
    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer')    
    parser.add_argument('--lr_ve', type=float, default=1e-4, help='the learning rate for the visual extractor.') # 1e-4 , 5e-5
    parser.add_argument('--lr_ed', type=float, default=5e-4, help='the learning rate for the remaining parameters.') # 5e-4, 1e-4
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay')
    parser.add_argument('--adam_eps', type=float, default=1e-9, help='the weight decay')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler')
    parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler')

    # Others
    parser.add_argument('--exp_name', type=str, default='mesh_memory', help='.')
    # parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--seed', type=int, default=112233, help='.')
    # parser.add_argument('--resume', type=str, default = './results/iu_xray/v1_0.1_dropout_40_m_epoch_49.pth', help='whether to resume the training from existing checkpoints')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints')

    args = parser.parse_args()
    return args


def main():

    # with open('./data/iu_xray/pos_dict.json', 'r') as f:
        # pos_dict = json.load(f)
    
    # get_dict(pos_dict)

    # parse arguments
    
    args = parse_args()
    tokenizer = Tokenizer(args)
    # ll = [0 for _ in range(762)]
    # for key in pos_dict:
    #     ll[int(key)] = 1
    # for ele in range(1, 762):
    #     if str(ele) not in pos_dict:
    #         print(ele,':', tokenizer.get_token_by_id(ele))
    # exit()
    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # create tokenizer
    
    
    # create data loader
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    valid_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    # build model architecture
    model = MMTModel(args, tokenizer)
    # print(model)
    # exit()
    # model.eval()
    # traced_model = torch.jit.trace(model, torch.randn(8, 2, 3, 224, 224,dtype = torch.float))  # 示例输入
    # traced_model.save("model.pt")

    # exit()
    # get function handles of loss and metrics
    criterion = compute_loss
    # criterion = NLLLoss(ignore_index= 0)
    metrics = compute_scores

    # build optimizer, learning rate scheduler
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    # build trainer and start to train
    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler,
                      train_dataloader, valid_dataloader, test_dataloader)
    trainer.train()


if __name__ == '__main__':
    main()
