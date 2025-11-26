from __future__ import print_function
from cProfile import label
import os
import time
import yaml
import pprint
import random
import pickle
import shutil
import inspect
import argparse
from collections import OrderedDict, defaultdict

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
import apex

from utils import count_params, import_class
from model.ssnet_tools import SSNetLoss


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_parser():
    # parameter priority: command line > config file > default
    parser = argparse.ArgumentParser(description='MS-G3D')

    parser.add_argument(
        '--work-dir',
        type=str,
        # required=True,
        default='work_dir/ntu/xsub/msg3d_joint_test',
        help='the work folder for storing results')
    parser.add_argument('--model_saved_name', default='')
    parser.add_argument(
        '--config',
        # default='./config/nturgbd-cross-subject/train_joint.yaml',
        default='./config/nturgbd-cross-subject/train_joint_msg3d2.yaml',
        help='path to the configuration file')
    parser.add_argument(
        '--assume-yes',
        action='store_true',
        help='Say yes to every prompt')

    parser.add_argument(
        '--phase',
        default='train',
        help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    parser.add_argument(
        '--seed',
        type=int,
        default=random.randrange(200),
        help='random seed')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=1,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--eval-start',
        type=int,
        default=1,
        help='The epoch number to start evaluating models')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    parser.add_argument(
        '--feeder',
        default='feeder.feeder',
        help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=os.cpu_count(),
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    parser.add_argument(
        '--model',
        default=None,
        help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')
    parser.add_argument(
        '--half',
        action='store_true',
        help='Use half-precision (FP16) training')
    parser.add_argument(
        '--amp-opt-level',
        type=int,
        default=1,
        help='NVIDIA Apex AMP optimization level')

    parser.add_argument(
        '--base-lr',
        type=float,
        default=0.01,
        help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument(
        '--optimizer',
        default='SGD',
        help='type of optimizer')
    parser.add_argument(
        '--nesterov',
        type=str2bool,
        default=False,
        help='use nesterov or not')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='training batch size')
    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=256,
        help='test batch size')
    parser.add_argument(
        '--forward-batch-size',
        type=int,
        default=16,
        help='Batch size during forward pass, must be factor of --batch-size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--optimizer-states',
        type=str,
        help='path of previously saved optimizer states')
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='path of previously saved training checkpoint')
    parser.add_argument(
        '--debug',
        type=str2bool,
        default=False,
        help='Debug mode; default false')

    return parser


class Processor():
    """Processor for Skeleton-based Action Recgnition"""

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            # Added control through the command line
            arg.train_feeder_args['debug'] = arg.train_feeder_args['debug'] or self.arg.debug
            logdir = os.path.join(arg.work_dir, 'trainlogs')
            if not arg.train_feeder_args['debug']:
                # logdir = arg.model_saved_name
                if os.path.isdir(logdir):
                    print(f'log_dir {logdir} already exists')
                    if arg.assume_yes:
                        answer = 'y'
                    else:
                        answer = input('delete it? [y]/n:')
                    if answer.lower() in ('y', ''):
                        shutil.rmtree(logdir)
                        print('Dir removed:', logdir)
                    else:
                        print('Dir not removed:', logdir)

                self.train_writer = SummaryWriter(
                    os.path.join(logdir, 'train'), 'train')
                self.val_writer = SummaryWriter(
                    os.path.join(logdir, 'val'), 'val')
            else:
                self.train_writer = SummaryWriter(
                    os.path.join(logdir, 'debug'), 'debug')

        self.load_model()
        
        self.load_param_groups()
        self.load_optimizer()
        self.load_lr_scheduler()
        self.load_data()

        self.global_step = 0
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0

        if self.arg.half:
            self.print_log('*************************************')
            self.print_log('*** Using Half Precision Training ***')
            self.print_log('*************************************')
            self.model, self.optimizer = apex.amp.initialize(
                self.model,
                self.optimizer,
                opt_level=f'O{self.arg.amp_opt_level}'
            )
            if self.arg.amp_opt_level != 1:
                self.print_log(
                    '[WARN] nn.DataParallel is not yet supported by amp_opt_level != "O1"')

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.print_log(
                    f'{len(self.arg.device)} GPUs available, using DataParallel')
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device
                )

    def load_model(self):
        output_device = self.arg.device[0] if type(
            self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)

        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        shutil.copy2(os.path.join('.', __file__), self.arg.work_dir)

        device = torch.device(f"cuda:{output_device}" if torch.cuda.is_available() else "cpu")
        self.model = Model(**self.arg.model_args).cuda(device)
        self.loss = SSNetLoss(alpha=0.5).cuda(output_device)
        # self.loss = nn.CrossEntropyLoss().cuda(output_device)
        # self.reg_loss = nn.MSELoss().cuda(output_device)
        self.print_log(
            f'Model total number of params: {count_params(self.model)}')

        if self.arg.weights:
            try:
                self.global_step = int(self.arg.weights[:-3].split('-')[-1])
            except:
                print('Cannot parse global_step from model weights filename')
                self.global_step = 0

            self.print_log(f'Loading weights from {self.arg.weights}')
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            for w in self.arg.ignore_weights:
                if weights.pop(w, None) is not None:
                    self.print_log(f'Sucessfully Remove Weights: {w}')
                else:
                    self.print_log(f'Can Not Remove Weights: {w}')

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                self.print_log('Can not find these weights:')
                for d in diff:
                    self.print_log('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

    def load_param_groups(self):
        """
        Template function for setting different learning behaviour
        (e.g. LR, weight decay) of different groups of parameters
        """
        self.param_groups = defaultdict(list)

        for name, params in self.model.named_parameters():
            self.param_groups['other'].append(params)

        self.optim_param_groups = {
            'other': {'params': self.param_groups['other']}
        }

    def load_optimizer(self):
        params = list(self.optim_param_groups.values())
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                params,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                params,
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError(
                'Unsupported optimizer: {}'.format(self.arg.optimizer))

        # Load optimizer states if any
        if self.arg.checkpoint is not None:
            self.print_log(
                f'Loading optimizer states from: {self.arg.checkpoint}')
            self.optimizer.load_state_dict(torch.load(
                self.arg.checkpoint)['optimizer_states'])
            current_lr = self.optimizer.param_groups[0]['lr']
            self.print_log(f'Starting LR: {current_lr}')
            self.print_log(
                f'Starting WD1: {self.optimizer.param_groups[0]["weight_decay"]}')
            if len(self.optimizer.param_groups) >= 2:
                self.print_log(
                    f'Starting WD2: {self.optimizer.param_groups[1]["weight_decay"]}')

    def load_lr_scheduler(self):
        self.lr_scheduler = MultiStepLR(
            self.optimizer, milestones=self.arg.step, gamma=0.1)
        if self.arg.checkpoint is not None:
            scheduler_states = torch.load(self.arg.checkpoint)[
                'lr_scheduler_states']
            self.print_log(
                f'Loading LR scheduler states from: {self.arg.checkpoint}')
            self.lr_scheduler.load_state_dict(scheduler_states)
            self.print_log(
                f'Starting last epoch: {scheduler_states["last_epoch"]}')
            self.print_log(
                f'Loaded milestones: {scheduler_states["last_epoch"]}')

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()

        def worker_seed_fn(worker_id):
            return init_seed(self.arg.seed + worker_id + 1)

        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=worker_seed_fn)

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=worker_seed_fn)

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open(os.path.join(self.arg.work_dir, 'config.yaml'), 'w') as f:
            yaml.dump(arg_dict, f)

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log(f'Local current time: {localtime}')

    def print_log(self, s, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            s = f'[ {localtime} ] {s}'
        print(s)
        if self.arg.print_log:
            with open(os.path.join(self.arg.work_dir, 'log.txt'), 'a') as f:
                print(s, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def save_states(self, epoch, states, out_folder, out_name):
        out_folder_path = os.path.join(self.arg.work_dir, out_folder)
        out_path = os.path.join(out_folder_path, out_name)
        os.makedirs(out_folder_path, exist_ok=True)
        torch.save(states, out_path)

    def save_checkpoint(self, epoch, out_folder='checkpoints', filename=None):
        state_dict = {
            'epoch': epoch,
            'optimizer_states': self.optimizer.state_dict(),
            'lr_scheduler_states': self.lr_scheduler.state_dict(),
        }

        # Use the provided filename if given, else fallback to default
        checkpoint_name = filename or f'checkpoint-{epoch}-fwbz{self.arg.forward_batch_size}-{int(self.global_step)}.pt'
        self.save_states(epoch, state_dict, out_folder, checkpoint_name)

    def save_weights(self, epoch, out_folder='weights', filename=None):
        state_dict = self.model.state_dict()
        weights = OrderedDict([
            [k.split('module.')[-1], v.cpu()]
            for k, v in state_dict.items()
        ])

        # Use the provided filename if given, else fallback to default
        weights_name = filename or f'weights-{epoch}-{int(self.global_step)}.pt'
        self.save_states(epoch, weights, out_folder, weights_name)

    def train(self, epoch, save_model=False):
        self.model.train()
        process = tqdm(self.data_loader['train'], dynamic_ncols=True)
        loss_values = []
        total_loss = total_cls = total_reg = 0.0

        if ((epoch + 1) % 1 == 0):
            log_file = os.path.join(self.arg.work_dir, f'train_epoch{epoch+1}.txt')
            f_log = open(log_file, 'w')
        else:
            f_log = None

        for batch_idx, (batch_data, batch_label, distance, total_len, index) in enumerate(process):
            self.global_step += 1

            with torch.no_grad():
                batch_data = batch_data.float().cuda(self.output_device)
                batch_label = batch_label.long().cuda(self.output_device)
                distance = distance.float().cuda(self.output_device)

            self.optimizer.zero_grad()

            # batch_data = batch_data.float().to(self.output_device)       # (B, T, C)
            # batch_label = batch_label.long().to(self.output_device)      # (B,)
            # distance = distance.float().to(self.output_device)           # (B,)

            distance_norm = distance / distance.max()

            logits, s_pred = self.model(batch_data, st_prev=distance_norm)

            if ((epoch + 1) % 1 == 0):
                for i in range(batch_data.size(0)):
                    pred_dist = s_pred[i].item()
                    pred_dist_dnorm = pred_dist * distance.max().item()
                    true_dist = distance[i].item()
                    pred_class = logits[i].argmax().item()
                    true_class = batch_label[i].item()
                    f_log.write(
                        f"Batch {batch_idx}, Sample {i}, "
                        f"s_pred: {pred_dist:.4f}, s_pred_dnorm: {pred_dist_dnorm:.4f}, "
                        f"s_gts: {true_dist:.4f}, class_pred: {pred_class}, class_gts: {true_class}\n"
                    )

            loss, cls_loss, reg_loss = self.loss(
                logits, s_pred, batch_label, distance_norm)
            
            self.optimizer.zero_grad()
            if self.arg.half:
                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            if self.lr_scheduler and hasattr(self.lr_scheduler, "step_per_batch"):
                self.lr_scheduler.step()

            loss_values.append(loss.item())
            total_loss += loss.item()
            total_cls += cls_loss.item()
            total_reg += reg_loss.item()

            process.set_description(
                f"(BS {batch_data.size(0)}) loss: {loss.item():.4f}")

        avg_loss = total_loss / len(loss_values)
        avg_cls = total_cls / len(loss_values)
        avg_reg = total_reg / len(loss_values)
        print(
            f"Epoch {epoch+1} — Avg loss: {avg_loss:.4f} | cls: {avg_cls:.4f} | reg: {avg_reg:.4f}")

        self.save_checkpoint(epoch + 1, out_folder='checkpoints',
                             filename=f'checkpoint-{epoch}-latest.pt')
        self.save_weights(epoch + 1, out_folder='weights',
                          filename=f'weights-{epoch}-latest.pt')

        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.best_acc_epoch = epoch + 1
            self.save_checkpoint(
                epoch + 1, out_folder='checkpoints', filename=f'checkpoint-{epoch}-best.pt')
            self.save_weights(epoch + 1, out_folder='weights',
                              filename=f'weights-{epoch}-best.pt')
            self.print_log(
                f"Best model at epoch {epoch + 1} (loss={avg_loss:.4f})")

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None, log_file=None):
        if epoch + 1 < self.arg.eval_start:
            return

        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')

        if ((epoch + 1) % 1 == 0):
            if log_file is None:
                log_file = os.path.join(self.arg.work_dir, f'eval_epoch{epoch+1}.txt')

            f_log = open(log_file, 'w')
        else:
            f_log = None
        
        with torch.no_grad():
            self.model = self.model.cuda(self.output_device)
            self.model.eval()
            self.print_log(f'Eval epoch: {epoch + 1}')
            for ln in loader_name:
                loss_values = []
                score_batches = []
                step = 0
                st_prev = None
                process = tqdm(self.data_loader[ln], dynamic_ncols=True)
                for batch_idx, (data, label, distance, total_len, index) in enumerate(process):
                    # print(data.shape, label, distance)
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    distance = distance.float().cuda(self.output_device)
                    total_len = total_len.long().cuda(self.output_device)
                    # print('label:', label, 'distance:', distance, 'total_len:', total_len)

                    logits, s_pred = self.model(data, st_prev=st_prev)
                    st_prev = s_pred

                    if ((epoch + 1) % 1 == 0):
                        for i in range(data.size(0)):
                            pred_dist = s_pred[i].item()
                            pred_dist_dnorm = pred_dist * distance.max().item()
                            true_dist = distance[i].item()
                            pred_class = logits[i].argmax().item()
                            true_class = label[i].item()
                            f_log.write(
                                f"Batch {batch_idx}, Sample {i}, "
                                f"pred_dist: {pred_dist:.4f}, pred_dist_dnorm: {pred_dist_dnorm:.4f}, "
                                f"true_dist: {true_dist:.4f}, pred_class: {pred_class}, true_class: {true_class}\n"
                            )

                    # s_pred = s_pred.squeeze() / distance.max()
                    distance = distance / distance.max()
                    # print('s_pred after norm:', s_pred, distance)
                    if isinstance(logits, tuple):
                        logits, l1 = logits
                        l1 = l1.mean()
                    else:
                        l1 = 0
                    loss, cls_loss, reg_loss = self.loss(
                        logits, s_pred, label, distance)
                    score_batches.append(logits.data.cpu().numpy())
                    loss_values.append(loss.item())

                    _, predict_label = torch.max(logits.data, 1)
                    step += 1

                    if wrong_file is not None or result_file is not None:
                        predict = list(predict_label.cpu().numpy())
                        true = list(label.data.cpu().numpy())
                        for i, x in enumerate(predict):
                            if result_file is not None:
                                f_r.write(str(x) + ',' + str(true[i]) + '\n')
                            if x != true[i] and wrong_file is not None:
                                f_w.write(
                                    str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')

            score = np.concatenate(score_batches)
            loss = np.mean(loss_values)
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1

            print('Accuracy: ', accuracy, ' model: ', self.arg.work_dir)
            if self.arg.phase == 'train' and not self.arg.debug:
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('loss_l1', l1, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

            score_dict = dict(
                zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log(
                f'\tMean {ln} loss of {len(self.data_loader[ln])} batches: {np.mean(loss_values)}.')
            for k in self.arg.show_topk:
                self.print_log(
                    f'\tTop {k}: {100 * self.data_loader[ln].dataset.top_k(score, k):.2f}%')

            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

        # Empty cache after evaluation
        torch.cuda.empty_cache()

    def start(self):
        if self.arg.phase == 'train':
            self.best_loss = float('inf') 
            self.best_acc = 0.0
            self.best_acc_epoch = 0
            self.print_log(f'Parameters:\n{pprint.pformat(vars(self.arg))}\n')
            self.print_log(
                f'Model total number of params: {count_params(self.model)}')
            self.global_step = self.arg.start_epoch * \
                len(self.data_loader['train']) / self.arg.batch_size
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                # save_model = ((epoch + 1) % self.arg.save_interval == 0) or (epoch + 1 == self.arg.num_epoch)
                self.train(epoch, save_model=False)
                # self.eval(epoch, save_score=self.arg.save_score,
                #           loader_name=['test'])
                if(epoch + 1) % 5 == 0:
                    self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])

                self.save_checkpoint(
                    epoch + 1, out_folder='checkpoints', filename=f'checkpoint-{epoch}-latest.pt')
                if self.best_acc_epoch == epoch + 1:
                    self.print_log(
                        f'Best model at epoch {epoch + 1} (acc={self.best_acc:.4f})')
                    self.save_checkpoint(
                        epoch + 1, out_folder='checkpoints', filename=f'checkpoint-{epoch}-best.pt')
                    self.save_weights(
                        epoch + 1, out_folder='weights', filename=f'weights-{epoch}-best.pt')

            num_params = sum(p.numel()
                             for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(
                f'Forward Batch Size: {self.arg.forward_batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')

        elif self.arg.phase == 'test':
            if not self.arg.test_feeder_args['debug']:
                wf = os.path.join(self.arg.work_dir, 'wrong-samples.txt')
                rf = os.path.join(self.arg.work_dir, 'right-samples.txt')
            else:
                wf = rf = f_log = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')

            self.print_log(f'Model:   {self.arg.model}')
            self.print_log(f'Weights: {self.arg.weights}')

            self.eval(
                epoch=0,
                save_score=self.arg.save_score,
                loader_name=['test'],
                wrong_file=wf,
                result_file=rf,
                log_file=f_log
            )

            self.print_log('Done.\n')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG:', k)
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()


if __name__ == '__main__':
    main()
