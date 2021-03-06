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
from models.model_sirta import SirtaModel #
from data.sirta.SirtaDataset import SirtaDataset
from metrics.regression import RegMetrics
from trainers.trainer_sirta_sets_creation import Sirta_seq_generator
from metrics.skill_scores import SkillScore
from utils import Config, Logger, format_time, print_model_spec, get_git_hash

class SirtaTrainer(Trainer):

    def create_data(self):

        sirta_sets_idx = Sirta_seq_generator(self.config.nb_training_seq, self.config.nb_validation_seq, self.config.lookback,
                                             self.config.lookforward, socket.gethostname(), self.config.preprocessed_dataset)

        self.training_seq_indexes, self.validation_seq_indexes = sirta_sets_idx.training_seq_indexes, sirta_sets_idx.validation_seq_indexes

        # Smart Persistence
        self.skill_score = SkillScore(self.config.shades, self.config.IMG_SIZE, self.config.lookback, self.config.lookforward, self.training_seq_indexes, self.validation_seq_indexes)

        self.train_dataset = SirtaDataset(self.training_seq_indexes, self.config.shades, self.config.IMG_SIZE, self.config.lookback, self.config.lookforward, self.config.preprocessed_dataset)
        self.val_dataset = SirtaDataset(self.validation_seq_indexes, self.config.shades, self.config.IMG_SIZE, self.config.lookback, self.config.lookforward, self.config.preprocessed_dataset)

        #self.train_dataset = SirtaDataset(mode='train')
        #self.val_dataset = SirtaDataset(mode='val')

        workers = self.config.n_workers

        if socket.gethostname() == 'mario':
            workers = 0

        self.train_dataloader = DataLoader(self.train_dataset, self.config.batch_size,
                                           num_workers=workers, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, self.config.batch_size,
                                         num_workers=workers, shuffle=False)

    def create_model(self):
        self.model = SirtaModel()

    def create_loss(self):
        self.loss_fn = nn.MSELoss() #CrossEntropyLoss()

    def create_optimiser(self):
        parameters_with_grad = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimiser = Adam(parameters_with_grad, self.config.learning_rate, weight_decay=self.config.weight_decay)

    def create_metrics(self):
        self.train_metrics = RegMetrics('train', self.tensorboard, self.session_name, self.skill_score)
        self.val_metrics = RegMetrics('val', self.tensorboard, self.session_name, self.skill_score)


    def forward_model(self, batch):
        return self.model(batch['images'], batch['aux_data'].float())

    def forward_loss(self, batch, output):
        #print('output : ', output)
        #print('batch[irradiance] : ', batch['irradiance'])
        return self.loss_fn(output, batch['irradiance'].float()) #.float()

    def visualise(self, batch, output, mode):
        self.tensorboard.add_images(mode + '/images_short', unsqueeze(batch['images'][:,0,:,:],1), self.global_step, dataformats='NCHW')
        self.tensorboard.add_images(mode + '/images_long', unsqueeze(batch['images'][:,1,:,:],1), self.global_step, dataformats='NCHW')

        #self.tensorboard.add_images(mode + '/images', batch_images, self.global_step, dataformats='NCHW')


#nb_training_seq = 1000
#nb_validation_seq = 200
#lookback = 1
#lookforward = 5

#shades = 'Y'
#IMG_SIZE = 128

#trainer = SirtaTrainer(shades, IMG_SIZE, nb_training_seq, nb_validation_seq, lookback, lookforward, options)
#trainer.create_data()