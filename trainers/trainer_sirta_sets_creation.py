from __future__ import print_function, division
import os
import torch
import pandas as pd
import cv2
import random
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

"""Same issue with import _C module"""
# from torchvision import transforms, utils
import socket
from time import sleep

from data.sirta.directories import data_images_dir, eumetsat_sat_images, data_sirta_grid, \
    data_clear_sky_irradiance_dir, data_irradiance_dir


class Sirta_seq_generator():

    def __init__(self, nb_training_seq, nb_validation_seq, lookback, lookforward, step, averaged_15min_dataset, helper,
                 computer=socket.gethostname(), preprocessed_dataset=True):
        self.nb_training_seq = nb_training_seq
        self.nb_validation_seq = nb_validation_seq
        self.lookback = lookback
        self.lookforward = lookforward
        self.computer = computer
        self.preprocessed_dataset = preprocessed_dataset
        self.step = step
        self.averaged_15min_dataset = averaged_15min_dataset
        self.helper = helper
        self.sat_images = False
        self.training_seq_indexes, self.validation_seq_indexes, self.mean, self.std = self.create_train_val_list(
            nb_training_seq,
            nb_validation_seq, lookback,
            lookforward, self.computer)

    def find_next_seq_index(self, y, m, d, h, minu, lookback, lookforward):  # return index of the next sequence
        # pos = lookback+lookforward+10+1
        # change when more samples
        pos = 1  # +lookback+lookforward+1

        id = self.helper.neighbours(y, m, d, h, minu, pos, True)
        return id

    def test_seq(self, y, m, d, h, minu, computer):  # check if every sample of the sequence exists

        if self.sat_images:
            DATADIR = eumetsat_sat_images(computer)
        else:
            DATADIR = data_sirta_grid(computer)

        ans = True
        lb_id = self.helper.lookback_indexes(y, m, d, h, minu)
        lf_id = self.helper.lookforward_index(y, m, d, h, minu)

        for y, m, d, h, minu in lb_id:
            Y, M, D, H, minut = self.helper.string_index(y, m, d, h, minu)

            folder_name = '{}'.format(Y)
            path = os.path.join(DATADIR, folder_name)
            if self.sat_images:
                if os.path.isdir(path):
                    file_name_1 = '{}{}/HRV/{}{}.jpg'.format(M, D, H, minut)
                    path_image_1 = os.path.join(path, file_name_1)
                    if os.path.isfile(path_image_1) != True:
                        ans = False
                    elif minu > 59:
                        ans = False
                else:
                    print('False path: ', path)
                    ans = False
            else:
                if os.path.isdir(path):
                    file_name_1 = 'Palaiseau_ghi_{}{}{}.csv'.format(Y, M, D)
                    path_image_1 = os.path.join(path, file_name_1)
                    if os.path.isfile(path_image_1) != True:
                        ans = False
                    elif minu > 59:
                        ans = False
                else:
                    print('False path: ', path)
                    ans = False

        return ans

    def get_mean_std(self, list):
        irradiances = []
        if self.averaged_15min_dataset:
            DATADIR_HELIO_IRRADIANCE = data_clear_sky_irradiance_dir(self.computer)
            helio_path = '{}{}'.format(DATADIR_HELIO_IRRADIANCE,
                                       'SoDa_HC3-METEO_lat48.713_lon2.209_2017-01-01_2018-12-31_1266955311.csv')
            helio_data = pd.read_csv(helio_path, header=34, usecols=['Date', 'Time', 'Global Horiz'])

            for [m, d, h, minu] in list:
                Y, M, D, H, Minu = self.helper.string_index(2018, m, d, h, minu)
                date_id = '{}-{}-{}'.format(Y, M, D)
                time_id = '{}:{}'.format(h, Minu)
                helio_data_irradiance_date = helio_data[helio_data['Date'] == date_id]
                helio_data_irradiance_time = helio_data_irradiance_date[helio_data_irradiance_date['Time'] == time_id]
                irradiance = float(helio_data_irradiance_time['Global Horiz']) * 4
                irradiances.append(irradiance)
        else:
            DATADIR_IRRADIANCE = data_irradiance_dir(self.computer)
            for [m, d, h, minu] in list:
                Y, M, D, H, Minu = self.helper.string_index(2018, m, d, h, minu)
                irra_file_name = '{}/solys2_radflux_{}{}{}.csv'.format(Y, Y, M, D)
                irra_path = os.path.join(DATADIR_IRRADIANCE, irra_file_name)
                df = pd.read_csv(irra_path, usecols=['global_solar_flux'])
                min_1440 = minu + 60 * h
                irradiance = df['global_solar_flux'][min_1440]
                irradiances.append(irradiance)

        mean = np.mean(irradiances)
        std = np.std(irradiances)

        return mean, std

    def create_train_val_list(self, nb_training_seq, nb_validation_seq, lookback, lookforward,
                              computer=socket.gethostname()):  # create the list of sequence indexes

        if self.sat_images:
            DATADIR = eumetsat_sat_images(computer)
        else:
            DATADIR = data_sirta_grid(computer)

        print('Number of Sequences : ', nb_training_seq)
        print('\n>> Generation of the list of sequence indexes')
        training_list = []
        validation_list = []
        irradiances = []
        mean = False
        random.seed(0.75)  #

        # dataset = '2017_2019'
        dataset = '2018'

        if dataset == '2017_2019':
            for Y in [2017]:
                if Y == 2017:
                    set_id = 'training_set'
                elif Y == 2018:
                    set_id = 'validation_set'

                for m in range(1, 13):  # range(1,13), m = month
                    if m <= 9:
                        M = '0{}'.format(m)
                    else:
                        M = '{}'.format(m)

                    for d in range(1, 32):  # d:day
                        if random.random() < 0.5:
                            set_id = 'training_set'
                        else:
                            set_id = 'validation_set'

                        if d <= 9:
                            D = '0{}'.format(d)
                        else:
                            D = '{}'.format(d)
                        folder_name = '{}'.format(Y)
                        path = os.path.join(DATADIR, folder_name)

                        if os.path.isdir(path):
                            h = 8
                            minu = random.randint(0, int(60 / self.step))
                            minu = int(self.step * minu)
                            while h < 19:

                                if self.test_seq(Y, m, d, h, minu, computer):

                                    #### To be removed
                                    # if random.random() < 0.9:
                                    #    set_id = 'training_set'
                                    # else:
                                    #    set_id = 'validation_set'
                                    ####
                                    if set_id == 'training_set':
                                        training_list.append([Y, m, d, h, minu])
                                    if set_id == 'validation_set':
                                        validation_list.append([Y, m, d, h, minu])
                                Y, m, d, h, minu = self.find_next_seq_index(Y, m, d, h, minu, lookback, lookforward)


        elif dataset == '2018':
            y = 2018
            # for m in range(2, 10):  # range(1,13), m = month
            # if you wanna use sky images, month has to be from 6 - 8 and days from 1 to 6
            for m in range(5, 9):  # range(1,13), m = month
                # if m <= 9:
                # M = '0{}'.format(m)
                # else:
                # M = '{}'.format(m)

                for d in range(1, 32):  # d:day
                    # if d <= 9:
                    # D = '0{}'.format(d)
                    # else:
                    # D = '{}'.format(d)
                    # folder_name = '2018{}{}'.format(M, D)
                    folder_name = '{}'.format(y)
                    path = os.path.join(DATADIR, folder_name)

                    if os.path.isdir(path) == True:
                        h = 9
                        # h = 12
                        minu = random.randint(0, int(60 / self.step) - 1)
                        # minu = 12
                        minu = int(self.step * minu)
                        minu = 0
                        while h < 18:
                            if self.test_seq(y, m, d, h, minu, computer):
                                """if random.random() < 0.8:
                                    set_id = 'training_set'
                                else:
                                    set_id = 'validation_set'"""
                                if m == 5 or m == 6 or m == 7:
                                    set_id = 'training_set'
                                if m == 8:
                                    set_id = 'validation_set'
                                if set_id == 'training_set':
                                    training_list.append([m, d, h, minu])
                                if set_id == 'validation_set':
                                    validation_list.append([m, d, h, minu])
                            y, m, d, h, minu = self.find_next_seq_index(y, m, d, h, minu, lookback, lookforward)

        #random.shuffle(training_list)
        #random.shuffle(validation_list)
        print('\nNumber of Sequences available given the constraints :',
              np.shape(training_list)[0] + np.shape(validation_list)[0])
        print('\nNumber of Sequences available in the training list :', np.shape(training_list)[0])
        print('Number of Sequences available in the validation list :', np.shape(validation_list)[0])

        training_seq_indexes = training_list[0:nb_training_seq]
        validation_seq_indexes = validation_list[0:nb_validation_seq]
        # for sequential in order - LSTM
        #training_seq_indexes = training_list[2756:2756 + nb_training_seq]
        #validation_seq_indexes = validation_list[689:689 + nb_validation_seq]

        print('\nNumber of Sequences in the training list :', len(training_seq_indexes))
        print('Number of Sequences in the validation list :', len(validation_seq_indexes))

        mean, std = self.get_mean_std(training_seq_indexes)
        print('\nTraining Sequences mean : {:0.2f}'.format(mean))
        print('Training Sequences standard deviation : {:0.2f}'.format(std))

        print('\nTraining and validation List of sequence indexes created')

        return training_seq_indexes, validation_seq_indexes, mean, std
