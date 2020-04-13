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

from data.sirta.directories import data_images_dir, data_preprocessed_images_dir


class Sirta_seq_generator():

    def __init__(self, nb_training_seq, nb_validation_seq, lookback, lookforward, computer=socket.gethostname(), preprocessed_dataset=True):
        self.nb_training_seq = nb_training_seq
        self.nb_validation_seq = nb_validation_seq
        self.lookback = lookback
        self.lookforward = lookforward
        self.computer = computer
        self.preprocessed_dataset = preprocessed_dataset
        self.training_seq_indexes, self.validation_seq_indexes = self.create_train_val_list(nb_training_seq, nb_validation_seq, lookback, lookforward, self.computer)



    def neighbours(self, y, m, d, h, minu, pos):
        pos = pos * 2

        H = h
        Minu = minu + pos
        change = True
        while change == True:
            # if minu+pos<=58 and minu+pos>=0:
            #    Minu = minu+pos
            #    H=h
            change = False
            if Minu > 58:
                Minu = Minu - 60
                H = H + 1  # h<20 in the dataset
                change = True
            elif Minu < 0:
                change = True
                Minu = Minu + 60
                H = H - 1
        Minu = int(Minu)
        H = int(H)
        M = m
        D = d
        Y = y
        return (Y, M, D, H, Minu)


    def lookback_indexes(self, y, m, d, h, minu, lookback):  # return list of past samples indexes
        list = []
        for k in range(-lookback, 1):
            id = self.neighbours(y, m, d, h, minu, k)
            list.append(id)
        return (list)


    def lookforward_index(self, y, m, d, h, minu, lookforward):  # return index of future sample
        id = self.neighbours(y, m, d, h, minu, lookforward)
        return (id)


    def find_next_seq_index(self, y, m, d, h, minu, lookback, lookforward):  # return index of the next sequence
        # pos = lookback+lookforward+10+1
        #change when more samples
        pos = 1  # +lookback+lookforward+1

        id = self.neighbours(y, m, d, h, minu, pos)
        return (id)


    def test_seq(self, y, m, d, h, minu, lookback, lookforward, computer, preprocessed_dataset):  # check if every sample of the sequence exists

        if self.preprocessed_dataset:
            DATADIR = data_preprocessed_images_dir(computer)
        else:
            DATADIR = data_images_dir(computer)

        ans = True
        lb_id = self.lookback_indexes(y, m, d, h, minu, lookback)
        lf_id = self.lookforward_index(y, m, d, h, minu, lookforward)

        for y, m, d, h, minu in lb_id:
            if m <= 9:
                M = '0{}'.format(m)
            else:
                M = '{}'.format(m)
            if d <= 9:
                D = '0{}'.format(d)
            else:
                D = '{}'.format(d)
            if h < 10:
                H = '0{}'.format(h)
            else:
                H = '{}'.format(h)
            if minu < 10:
                minut = '0{}'.format(minu)
            else:
                minut = '{}'.format(minu)
            folder_name = '{}/{}{}{}'.format(y, y, M, D)
            #folder_name = '{}{}{}'.format(y, M, D)
            path = os.path.join(DATADIR, folder_name)
            if os.path.isdir(path) == True:
                file_name_1 = '{}{}{}{}{}00_01.jpg'.format(y, M, D, H, minut)
                file_name_2 = '{}{}{}{}{}00_03.jpg'.format(y, M, D, H, minut)
                path_image_1 = os.path.join(path, file_name_1)
                path_image_2 = os.path.join(path, file_name_2)
                if os.path.isfile(path_image_1) != True or os.path.isfile(path_image_2) != True:
                    # print('False image path : ', path_image_1)
                    ans = False
            else:
                print('False path: ', path)
                ans = False
        return ans


    def create_train_val_list(self, nb_training_seq, nb_validation_seq, lookback, lookforward, computer=socket.gethostname()):  # create the list of sequence indexes

        if self.preprocessed_dataset == True:
            DATADIR = data_preprocessed_images_dir(computer)
        else:
            DATADIR = data_images_dir(computer)

        print('Number of Sequences : ', nb_training_seq)
        print('\n>> Generation of the list of sequence indexes')
        training_list = []
        validation_list = []
        random.seed(0.75) #

        #dataset = '2017_2019'
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
                        folder_name = '{}/{}{}{}'.format(Y, Y, M, D)
                        path = os.path.join(DATADIR, folder_name)


                        if os.path.isdir(path):
                            h = 9
                            minu = random.randint(0, 29)
                            minu = int(2 * minu)
                            while h < 19:

                                if self.test_seq(Y, m, d, h, minu, lookback, lookforward, computer, self.preprocessed_dataset):

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
            #for m in range(2, 10):  # range(1,13), m = month
            for m in range(6, 8):  # range(1,13), m = month
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
                    #folder_name = '2018{}{}'.format(M, D)
                    folder_name = '{}/{}{}{}'.format(y, y, M, D)
                    path = os.path.join(DATADIR, folder_name)

                    if os.path.isdir(path) == True:
                        h = 8
                        #h = 12
                        minu = random.randint(0, 29)
                        #minu = 12
                        minu = int(2 * minu)
                        while h < 19:
                            if self.test_seq(y, m, d, h, minu, lookback, lookforward, computer, self.preprocessed_dataset):

                                #### To be removed
                                if random.random() < 0.5:
                                   set_id = 'training_set'
                                else:
                                   set_id = 'validation_set'
                                ####
                                if set_id == 'training_set':
                                    training_list.append([m, d, h, minu])
                                if set_id == 'validation_set':
                                    validation_list.append([m, d, h, minu])
                            y, m, d, h, minu = self.find_next_seq_index(y, m, d, h, minu, lookback, lookforward)

        random.shuffle(training_list)
        random.shuffle(validation_list)
        print('\nNumber of Sequences available given the constraints :',
              np.shape(training_list)[0] + np.shape(validation_list)[0])
        print('\nNumber of Sequences available in the training list :', np.shape(training_list)[0])
        print('Number of Sequences available in the validation list :', np.shape(validation_list)[0])

        training_seq_indexes = training_list[0:nb_training_seq]
        validation_seq_indexes = validation_list[0:nb_validation_seq]

        print('\nNumber of Sequences in the training list :', len(training_seq_indexes))
        print('Number of Sequences in the validation list :', len(validation_seq_indexes))

        print('\nTraining and validation List of sequence indexes created')

        return training_seq_indexes, validation_seq_indexes