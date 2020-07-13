import os
import sys
import socket
import yaml
import datetime
import numpy as np

from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch import unsqueeze

from trainers.trainer import Trainer
from models.model_sirta import SirtaModel
from models.model_lstm import LSTMModel
from data.sirta.SirtaDataset import SirtaDataset
from metrics.regression import RegMetrics
from trainers.trainer_sirta_sets_creation import Sirta_seq_generator
from metrics.skill_scores import SkillScore
from utils import Config, Logger, format_time, print_model_spec, get_git_hash


class SirtaTrainer(Trainer):

    def create_data(self):
        sirta_sets_idx = Sirta_seq_generator(self.config.nb_training_seq, self.config.nb_validation_seq,
                                             self.config.lookback, self.config.lookforward, self.config.step, self.config.averaged_15min_dataset,
                                             self.helper, socket.gethostname(), self.config.preprocessed_dataset, self.config.sat_images)

        self.training_seq_indexes, self.validation_seq_indexes, self.mean, self.std = sirta_sets_idx.training_seq_indexes, \
                                                                 sirta_sets_idx.validation_seq_indexes, sirta_sets_idx.mean, sirta_sets_idx.std

        # Smart Persistence
        self.skill_score = SkillScore(self.config.shades, self.config.IMG_SIZE, self.config.lookback,
                                      self.config.lookforward, self.training_seq_indexes, self.validation_seq_indexes,
                                      self.config.step, self.std, self.helper)

        self.train_dataset = SirtaDataset(self.training_seq_indexes, self.config.shades, self.config.IMG_SIZE,
                                          self.config.lookback, self.config.lookforward, self.config.step,
                                          self.config.averaged_15min_dataset, self.mean, self.std, self.helper,
                                          self.config.preprocessed_dataset, self.config.sat_images)
        self.val_dataset = SirtaDataset(self.validation_seq_indexes, self.config.shades, self.config.IMG_SIZE,
                                        self.config.lookback, self.config.lookforward, self.config.step,
                                        self.config.averaged_15min_dataset, self.mean, self.std, self.helper,
                                        self.config.preprocessed_dataset, self.config.sat_images)

        # self.train_dataset = SirtaDataset(mode='train')
        # self.val_dataset = SirtaDataset(mode='val')

        workers = self.config.n_workers

        if socket.gethostname() == 'mario':
            workers = 0

        self.train_dataloader = DataLoader(self.train_dataset, self.config.batch_size,
                                           num_workers=workers, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, self.config.batch_size,
                                         num_workers=workers, shuffle=False)

    def mean_std(self):
        return self.mean, self.std

    def create_model(self, lstm=False, nowcast=False, image_type='RGB_HRV'):
        #lstm = True
        nowcast = True
        if image_type == 'HRV':
            channels = 1
        if image_type == 'RGB':
            channels = 3
        if image_type == 'RGB_HRV':
            channels = 4
        if not lstm and not nowcast:
            self.model = SirtaModel(self.config.lookback + 1)
        elif not lstm and nowcast:
            self.model = SirtaModel(channels)
        else:
            self.model = LSTMModel()
            #self.model.reset_hidden_state()

    def create_loss(self):
        self.loss_fn = nn.MSELoss()  # CrossEntropyLoss()

    def create_optimiser(self):
        parameters_with_grad = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimiser = Adam(parameters_with_grad, self.config.learning_rate, weight_decay=self.config.weight_decay)

    def create_metrics(self):
        self.train_metrics = RegMetrics('train', self.tensorboard, self.session_name, self.skill_score, self.std)
        self.val_metrics = RegMetrics('val', self.tensorboard, self.session_name, self.skill_score, self.std)

    def forward_model(self, batch):
        return self.model(batch['images'], batch['aux_data'].float())

    def forward_loss(self, batch, output):
        # print('output : ', output)
        # print('batch[irradiance] : ', batch['irradiance'])
        return self.loss_fn(output, batch['irradiance'].float())  # .float()

    def visualise(self, batch, output, mode):
        self.tensorboard.add_images(mode + '/images_short', unsqueeze(batch['images'][:, 0, :, :], 1), self.global_step,
                                    dataformats='NCHW')
        # self.tensorboard.add_images(mode + '/images_long', unsqueeze(batch['images'][:,1,:,:],1), self.global_step, dataformats='NCHW')

        # self.tensorboard.add_images(mode + '/images', batch_images, self.global_step, dataformats='NCHW')

    def create_sample(self):
        self.dataset = SirtaDataset(self.index, self.config.shades, self.config.IMG_SIZE, self.config.lookback,
                                    self.config.lookforward, self.config.step, self.config.averaged_15min_dataset, self.mean, self.std,
                                    self.helper, self.config.preprocessed_dataset)
        self.sample = SirtaDataset.get_image(self.dataset, self.index)

        return self.sample
