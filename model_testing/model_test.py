import os
import sys
import datetime
import socket
from abc import ABCMeta, abstractmethod
from time import time
from data.sirta.SirtaDataset import show_data_batch

import yaml
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter

from utils import Config, Logger, format_time, print_model_spec, get_git_hash


class Model_Test:
    __meta_class__ = ABCMeta

    def __init__(self, options):
        self.options = options

        ##########
        # Trainer utils
        ##########
        self.global_step = 0
        self.start_time = None
        self.best_score = -float('inf')

        ##########
        # Initialise/restore session
        ##########
        self.config = None
        self.session_name = ''
        if self.options.config:
            self.initialise_session()
        elif self.options.restore:
            self.restore_session()
        else:
            raise ValueError('Must specify --config or --restore path.')

        self.tensorboard = SummaryWriter(self.session_name, comment=self.config.tag)

        if self.config.gpu_nb == None or socket.gethostname() == 'windle':
            device_name = 'cuda'
        else:
            device_name = 'cuda:{}'.format(self.config.gpu_nb)

        self.device = torch.device(device_name) if self.config.gpu else torch.device('cpu')

        ##########
        # Sample
        ##########
        self.dataset = None
        self.sample = None
        self.index = None


        ##########
        # Model
        ##########
        self.model = None
        self.create_model()
        self.model.to(self.device)

        ##########
        # Loss
        ##########
        self.loss_fn = None
        self.create_loss()

        ##########
        # Optimiser
        ##########
        self.optimiser = None
        self.create_optimiser()

        ##########
        # Metrics
        ##########
        self.train_metrics = None
        self.val_metrics = None
        #self.create_metrics()

        # Restore model
        if self.options.restore:
            self.load_checkpoint()


    @abstractmethod
    def create_sample(self):
        """Create train/val datasets and dataloaders."""

    @abstractmethod
    def create_model(self):
        """Build the neural network."""

    @abstractmethod
    def create_loss(self):
        """Build the loss function."""

    @abstractmethod
    def create_optimiser(self):
        """Create the model's optimiser."""

    @abstractmethod
    def create_metrics(self):
        """Implement the metrics."""

    @abstractmethod
    def forward_model(self, batch):
        """Compute the output of the model."""

    @abstractmethod
    def forward_loss(self, batch, output):
        """Compute the loss."""

    @abstractmethod
    def visualise(self, batch, output, mode):
        """Visualise inputs and outputs on tensorboard."""


    def test_step(self, batch, iteration):
        # move batch to device
        self.preprocess_batch(batch)
        # foward_model computes model out put for given input
        output = self.forward_model(batch)
        # forward_loss computes the loss for that output
        loss = self.forward_loss(batch, output)

        # metrics
        self.val_metrics.update(output, batch['irradiance'])
        if iteration == 0:
            self.visualise(batch, output, 'val')

        # return loss
        return loss.item()

    # validation print out
    def test(self):
        self.index = [7, 12, 13, 15]
        self.sample = self.create_sample()
        print('Images size: ', self.sample['images'].size())
        print('Aux_data size: ', self.sample['aux_data'].size())
        print('irradiance size: ', self.sample['irradiance'].size())
        forecast = self.forward_model(self.sample) * 288.8 + 434.4
        actual = self.sample['irradiance'][0] * 288.8 + 434.4
        print(forecast.item(), actual.item())


    def print_log(self, loss, step_duration, data_fetch_time, model_update_time):
        """Print a log statement to the terminal."""
        samples_per_sec = self.config.batch_size / step_duration
        time_so_far = time() - self.start_time
        training_time_left = (self.config.n_iterations / self.global_step - 1.0) * time_so_far
        print_string = 'Iteration {:>6}/{} | examples/s: {:5.1f}' + \
                       ' | loss: {:.4f} | time elapsed: {} | time left: {}'
        print(print_string.format(self.global_step, self.config.n_iterations, samples_per_sec,
                                  loss, format_time(time_so_far), format_time(training_time_left)))
        print('Fetch data time: {:.0f}ms, model update time: {:.0f}ms\n'.format(1000 * data_fetch_time,
                                                                                1000 * model_update_time))

    def save_checkpoint(self):
        checkpoint = dict(model=self.model.state_dict(),
                          optimiser=self.optimiser.state_dict(),
                          global_step=self.global_step,
                          best_score=self.best_score,
                          )

        checkpoint_name = os.path.join(self.session_name, 'checkpoint')
        torch.save(checkpoint, checkpoint_name)
        print('Model saved to: {}\n'.format(checkpoint_name))

    def load_checkpoint(self):
        checkpoint_name = os.path.join(self.session_name, 'checkpoint')
        map_location = 'cuda' if self.config.gpu else 'cpu'
        checkpoint = torch.load(checkpoint_name, map_location=map_location)

        self.model.load_state_dict(checkpoint['model'])
        self.optimiser.load_state_dict(checkpoint['optimiser'])
        self.global_step = checkpoint['global_step']
        self.best_score = checkpoint['best_score']
        print('Loaded model and optimiser weights from {}\n'.format(checkpoint_name))

    def _get_next_batch(self):
        if self._train_dataloader_iter is None:
            self._train_dataloader_iter = iter(self.train_dataloader)
        batch = None
        while batch is None:
            try:
                batch = next(self._train_dataloader_iter)
            except StopIteration:
                self._train_dataloader_iter = iter(self.train_dataloader)
        return batch

    def preprocess_batch(self, batch):
        # Cast to device
        for key, value in batch.items():
            batch[key] = value.to(self.device)

    def initialise_session(self):
        config_path = self.options.config
        # Load config file, save it to the experiment output path, and convert to a Config class.
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.session_name = self.create_session_name()
        self.config['session_name'] = self.session_name
        with open(os.path.join(self.session_name, 'config.yml'), 'w') as f:
            yaml.dump(self.config, f)
        self.config = Config(self.config)

        # Save git hash
        #with open(os.path.join(self.session_name, 'git_hash'), 'w') as f:
        #    f.write(get_git_hash() + '\n')

        # Save terminal outputs
        sys.stdout = Logger(os.path.join(self.session_name, 'logs.txt'))

    def restore_session(self):
        config_path = os.path.join(self.options.restore, 'config.yml')
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.config = Config(self.config)
        self.session_name = self.config.session_name

        # Compare git hash
        # Comment out for now
        """current_git_hash = get_git_hash()
        with open(os.path.join(self.session_name, 'git_hash')) as f:
            previous_git_hash = f.read().splitlines()[0]
        if current_git_hash != previous_git_hash:
            print('Restoring model with a different git hash.')
            print(f'Previous: {previous_git_hash}')
            print(f'Current: {current_git_hash}\n')"""

        # Save terminal outputs
        sys.stdout = Logger(os.path.join(self.session_name, 'logs.txt'))

    def create_session_name(self):
        now = datetime.datetime.now()
        session_name = 'session_{}_{:04d}_{:02d}_{:02d}_{:02d}_{:02d}_{:02d}_{}'.format(
            socket.gethostname(), now.year, now.month, now.day, now.hour, now.minute, now.second, self.config['tag'])
        session_name = os.path.join(self.config['output_path'], session_name)
        os.makedirs(session_name)
        return session_name

    def print_model(self):
        print(self.model)