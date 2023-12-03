# -*- coding: utf-8 -*-
import os
import logging
from abc import abstractmethod

from numpy import inf
import torch
import json
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange

# from modules.base_cmn import set_flag


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler):
        
        
        self.args = args
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged information into log dict
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)

            # print logged information to the screen
            for key, value in log.items():
                self.logger.info('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save the best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode=='min' and log[self.mnt_metric]<=self.mnt_best) or \
                                (self.mnt_mode=='max' and log[self.mnt_metric]>=self.mnt_best)
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. Model performance monitoring is disabled."
                            .format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. Training stops."
                                      .format(self.early_stop))
                    # break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _record_best(self, log):
        # improved_valid = (self.mnt_mode=='min' and log[self.mnt_metric]<=self.best_recorder['val'][self.mnt_metric]) or \
        #                   (self.mnt_mode=='max' and log[self.mnt_metric]>=self.best_recorder['val'][self.mnt_metric])
        # if improved_valid:
        #     self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode=='min' and log[self.mnt_metric_test]<=self.best_recorder['test'][self.mnt_metric_test]) or \
                        (self.mnt_mode=='max' and log[self.mnt_metric_test]>=self.best_recorder['test'][self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        self.logger.info('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            self.logger.info('\t{:15s}: {}'.format(str(key), value))

        self.logger.info('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            self.logger.info('\t{:15s}: {}'.format(str(key), value))

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        print('n_gpu:',n_gpu)
        if n_gpu_use>0 and n_gpu==0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine."
                    .format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use>0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, f'{self.args.exp_name}_epoch_%d.pth'%epoch)
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {}...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {}...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler,
                 train_dataloader, valid_dataloader, test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args, lr_scheduler)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        try:
            os.mkdir(f'./loss/{args.exp_name}')
        except:
            print('dir already exist!')
        self.writer = SummaryWriter(log_dir=f'./loss/{args.exp_name}', filename_suffix='events.out.tfevents.1699784221.DESKTOP-IFIR9G2.7060.0')
        # self.writer = SummaryWriter()
        
    def _train_epoch(self, epoch):
        log = {}
        # if (epoch - 1) % 4 == 0:
        # self._resume_checkpoint(f'./results/iu_xray/checkpoint_epoch_{epoch}.pth')
        self.logger.info('[{}/{}] Start to train in the training set.'.format(epoch, self.epochs))
        train_loss = 0
        self.model.train()
        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):
            # print(reports_ids)
        # tuple: len=batch_size "CXR2384_IM-0942"等, [batch_size, image_num, 3, 224, 224]
        # [batch_size, max_seq_len], [batch_size, max_seq_len]
            # break
            # print('reports_', reports_ids[0])
            images, reports_ids = images.to(self.device), reports_ids.to(self.device)
            reports_masks = reports_masks.to(self.device)
            # print(images.type())
            # exit()
            output = self.model(images, reports_ids, mode='train') # [batch_size, max_seq_len-1, vocab_size+1]
            # output = output.view(-1, output.size(-1))
            # print(reports_ids.size())
            # ind = tuple(reports_ids)
            # gt[ind - 1] = 1.
            # print(gt)
            # print(output.size())
            # print(gt.size())
            # exit()
            loss = self.criterion(output, reports_ids, reports_masks) # 6~7
            # gt = reports_ids[:, 1:]
            # gt = gt.reshape(output.size(0))
            # print(gt.size())
            # print(output.size())
            # loss = self.criterion(output, gt)
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.args.log_period == 0:
                # report = torch.argmax(output, -1)
                # reports = self.model.tokenizer.decode_batch(report.detach().cpu().numpy())                
                self.logger.info('Epoch: {}/{}, Step: {}/{}, Training Loss: {:.5f}.'
                                  .format(epoch, self.epochs, batch_idx, len(self.train_dataloader),
                                          train_loss / (batch_idx + 1)))
            # self.writer.add_graph(self.model, (images,reports_ids))
        #     break
        # self.logger.info('train_loss': train_loss / len(self.train_dataloader))
        
        self.writer.add_scalar('Loss/Train', train_loss / len(self.train_dataloader), epoch)
        

        # self.logger.info('[{}/{}] Start to evaluate in the validation set.'.format(epoch, self.epochs))
        # self.model.eval()
        # with torch.no_grad():
        #     valid_gts, valid_res = [], []
        #     for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.valid_dataloader):
        #         # break
        #         images, reports_ids = images.to(self.device), reports_ids.to(self.device)
        #         reports_masks = reports_masks.to(self.device)
        #         output, _ = self.model(images, mode='sample')
        #         reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
        #         ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
        #         valid_res.extend(reports)
        #         valid_gts.extend(ground_truths)
                # break
            # print('valid_gts',valid_gts)
            # print('valid_gts',valid_gts.size())
            # print('valid_ret',valid_res)
            # print('valid_ret',valid_ret.size())
            # print()
                # print('ground_truths:', len(ground_truths))
                
            # valid_met = self.metric_ftns({i: [gt] for i, gt in enumerate(valid_gts)},
            #                              {i: [re] for i, re in enumerate(valid_res)})
            # # print(valid_met)
            # # exit()
            # log.update(**{'val_' + k: v for k, v in valid_met.items()})

        self.logger.info('[{}/{}] Start to evaluate in the test set.'.format(epoch, self.epochs))
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            # expert = ['CXR3030_IM-1405', 'CXR2686_IM-1158', 'CXR458_IM-2089', 'CXR3430_IM-1659']
            # set_flag.set_epoch(epoch)
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                # sent = []
                # for ids in expert:
                    # try:
                        # sent_id = images_id.index(ids)
                        # sent.append(sent_id)
                #         # print(sent)
                    # except:
                        # pass
                # for ss in sent:
                    # expert.remove(images_id[ss])
                    # print(expert)
                    
                # if len(sent) != 0:
                #     set_flag.set_image_id(sent, images_id)
                    
                images, reports_ids = images.to(self.device), reports_ids.to(self.device)
                reports_masks = reports_masks.to(self.device)

                output, _ = self.model(images, mode='sample') # [batch_size, max_seq_len-1, vocab_size+1], [batch_size, 2, 3, 224, 224]
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                # set_flag.set_image_id()
                # break
            # !!! print generated and ground-truth reports
            f_res = open(f'records/{self.args.exp_name}_{epoch}.txt', 'w')
            if len(test_res) == 0:
                print(output)
            for line in test_res:
                f_res.write(line+'\n')
            f_res.close()
            #f_gts = open('records/truths.txt', 'w')
            #for line in test_gts:
            #    f_gts.write(line+'\n')
            #f_gts.close()
            # dic = {}
            # dic['gts'] = test_gts
            # dic['best'] = test_res
            # with open('./reports_record.json', 'w') as f:
            #     json.dump(dic, f)
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            #print(test_met)
            #exit()
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            for k, v in test_met.items():
                self.writer.add_scalar(f'{k}/epoch', v, epoch)

        self.lr_scheduler.step()
    # else:
    #     pass
        return log

