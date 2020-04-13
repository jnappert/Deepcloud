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
""" This is giving me error: from torchvision import _C
ImportError: DLL load failed: The specified module could not be found.
In sirtadataset, model_sirta, trainer_sirta_sets_creation"""
# from torchvision import transforms, utils
import socket

from data.sirta.directories import data_images_dir, data_irradiance_dir, data_preprocessed_images_dir

# Ignore warnings
#import warnings
#warnings.filterwarnings("ignore")

#plt.ion()   # interactive mode


class SirtaDataset(Dataset):
    """Sirta dataset."""

    def __init__(self, seq_indexes, shades, IMG_SIZE, lookback, lookforward, preprocessed_dataset=True, transform=None):
    #def __init__(self, computer, nb_training_seq, lookback, lookforward, mode=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with measurement.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        #self.measurements_list = pd.read_csv(csv_file)
        #self.root_dir = root_dir
        #self.nb_training_seq = nb_training_seq

        self.seq_indexes = seq_indexes
        self.shades = shades
        self.IMG_SIZE = IMG_SIZE
        self.lookback = lookback
        self.lookforward = lookforward
        self.computer = socket.gethostname()
        self.transform = transform
        self.preprocessed_dataset = preprocessed_dataset

        #self.training_seq_idexes = []
        #self.validation_seq_idexes = []
        #self.mode = mode


    def __len__(self):
        return len(self.seq_indexes) #int(nb_training_seq*0.8) #len(self.measurements_list)


    def __getitem__(self, idx):

        if self.preprocessed_dataset == True:
            DATADIR = data_preprocessed_images_dir(self.computer)
        else:
            DATADIR = data_images_dir(self.computer)

        DATADIR_IRRADIANCE = data_irradiance_dir(self.computer)

        #y, m, d, h, minu = self.seq_indexes[idx]

        y = 2018
        m, d, h, minu = self.seq_indexes[idx]
        samples_list_indexes = lookback_indexes(y, m, d, h, minu, self.lookback)
        past_images = []
        past_irradiances = []

        for [y, m, d, h, minu] in  samples_list_indexes:
            if m <= 9: M = '0{}'.format(m)
            else: M = '{}'.format(m)

            if d <= 9: D = '0{}'.format(d)
            else: D = '{}'.format(d)

            if h < 10: H = '0{}'.format(h)
            else: H = '{}'.format(h)

            if minu < 10: Minu = '0{}'.format(minu)
            else: Minu = '{}'.format(minu)

            folder_name = '{}/{}{}{}'.format(y, y, M, D)
            path = os.path.join(DATADIR, folder_name)


            if os.path.isdir(path) == True:

                irra_file_name = '{}/solys2_radflux_{}{}{}.csv'.format(y, y, M, D)
                irra_path = os.path.join(DATADIR_IRRADIANCE, irra_file_name)
                df = pd.read_csv(irra_path)

                min_1440 = minu + 60 * h
                irradiance = df['global_solar_flux'][min_1440]
                solar_zenith_angle = df['solar_zenith_angle'][min_1440]
                solar_azimuthal_angle = df['solar_azimuthal_angle'][min_1440]

                ### Zenith Angle
                solar_zenith_angle_rad = np.pi*solar_zenith_angle/180
                solar_zenith_angle_cos = np.cos(solar_zenith_angle_rad)
                solar_zenith_angle_sin = np.sin(solar_zenith_angle_rad)

                ### Azimutal Angle
                solar_azimuthal_angle_rad = np.pi*solar_azimuthal_angle/180
                solar_azimuthal_angle_cos = np.cos(solar_azimuthal_angle_rad)
                solar_azimuthal_angle_sin = np.sin(solar_azimuthal_angle_rad)


                file_name = [m, d, h, minu]  # '2018{}{}{}{}'.format(M,D, H, minu)

                # Target:
                y_target, m_target, d_target, h_target, minu_target = lookforward_index(y, m, d, h, minu, self.lookforward)
                min_1440_target = minu_target + 60 * h_target
                target = df['global_solar_flux'][min_1440_target]

                mean_irradiance = 434.4
                std_irradiance = 288.8

                target = (target - mean_irradiance) / std_irradiance
                irradiance = (irradiance - mean_irradiance) / std_irradiance
                past_irradiances.append(irradiance)

                aux_data = [solar_zenith_angle_rad, solar_zenith_angle_cos, solar_zenith_angle_sin,
                                solar_azimuthal_angle_rad, solar_azimuthal_angle_cos, solar_azimuthal_angle_sin]

                metadata_only = None
                if metadata_only == True:
                    aux_data = [solar_zenith_angle_rad, solar_zenith_angle_cos, solar_zenith_angle_sin,
                                solar_azimuthal_angle_rad, solar_azimuthal_angle_cos, solar_azimuthal_angle_sin,
                                irradiance]

                else:
                    file_name_1 = '{}{}{}{}{}00_01.jpg'.format(y, M, D, H, Minu)
                    file_name_2 = '{}{}{}{}{}00_03.jpg'.format(y, M, D, H, Minu)
                    path_image_1 = os.path.join(path, file_name_1)
                    path_image_2 = os.path.join(path, file_name_2)

                    if os.path.isfile(path_image_1) == True and os.path.isfile(path_image_2) == True:

                        if self.preprocessed_dataset:

                            new_array_1 = cv2.imread((path_image_1), cv2.IMREAD_GRAYSCALE)
                            new_array_2 = cv2.imread((path_image_2), cv2.IMREAD_GRAYSCALE)

                            new_array_1[new_array_1 == 0] = 1
                            new_array_2[new_array_2 == 0] = 1

                            new_array_1 = np.array(new_array_1).reshape(1, self.IMG_SIZE, self.IMG_SIZE)
                            new_array_2 = np.array(new_array_2).reshape(1, self.IMG_SIZE, self.IMG_SIZE)


                        else:
                            if self.shades == 'RGB' or self.shades == 'Bs' or self.shades == 'RBR':
                                img_array_BGR_1 = cv2.imread(path_image_1, 1)  # , cv2.IMREAD_GRAYSCALE)
                                img_array_1 = cv2.cvtColor(img_array_BGR_1, cv2.COLOR_BGR2RGB)
                                img_array_BGR_2 = cv2.imread(path_image_2, 1)  # , cv2.IMREAD_GRAYSCALE)
                                img_array_2 = cv2.cvtColor(img_array_BGR_2, cv2.COLOR_BGR2RGB)


                            elif self.shades == 'Y':
                                img_array_1 = cv2.imread((path_image_1), cv2.IMREAD_GRAYSCALE)
                                img_array_2 = cv2.imread((path_image_2), cv2.IMREAD_GRAYSCALE)

                            img_array_redim_1 = img_array_1[35:725, 150:860]
                            img_array_redim_2 = img_array_2[35:725, 150:860]

                            new_array_1 = cv2.resize(img_array_redim_1, (self.IMG_SIZE, self.IMG_SIZE))
                            new_array_2 = cv2.resize(img_array_redim_2, (self.IMG_SIZE, self.IMG_SIZE))

                            new_array_1[new_array_1 == 0] = 1
                            new_array_2[new_array_2 == 0] = 1

                            if self.shades == 'RGB':
                                new_array_1 = np.array(new_array_1).reshape(self.IMG_SIZE, self.IMG_SIZE, 3)
                                new_array_2 = np.array(new_array_2).reshape(self.IMG_SIZE, self.IMG_SIZE, 3)

                            elif self.shades == 'Bs':
                                new_array_1 = np.array(new_array_1).reshape(self.IMG_SIZE, self.IMG_SIZE, 3)
                                new_array_1 = np.sqrt(new_array_1[:, :, 1] ** 2 + new_array_1[:, :, 2] ** 2)
                                new_array_1 = np.array(new_array_1).reshape(self.IMG_SIZE, self.IMG_SIZE, 1)

                                new_array_2 = np.array(new_array_2).reshape(self.IMG_SIZE, self.IMG_SIZE, 3)
                                new_array_2 = np.sqrt(new_array_2[:, :, 1] ** 2 + new_array_2[:, :, 2] ** 2)
                                new_array_2 = np.array(new_array_2).reshape(self.IMG_SIZE, self.IMG_SIZE, 1)

                            elif self.shades == 'Y':
                                new_array_1 = np.array(new_array_1).reshape(1, self.IMG_SIZE, self.IMG_SIZE)
                                new_array_2 = np.array(new_array_2).reshape(1, self.IMG_SIZE, self.IMG_SIZE)

                            elif self.shades == 'RBR':
                                new_array_1 = np.array(new_array_1).reshape(self.IMG_SIZE, self.IMG_SIZE, 3)
                                new_array_1[:, :, 2][new_array_1[:, :, 2] <= 1.0 / 256.0] = 1.0 / 256.0
                                new_array_1 = np.divide(new_array_1[:, :, 0], new_array_1[:, :, 2])

                                bar = 0.4
                                new_array_1[new_array_1 < bar] = bar
                                new_array_1 = new_array_1 - bar
                                th = 1.1 * (1 - bar)
                                new_array_1[new_array_1 > th] = th
                                new_array_1 = new_array_1 / th

                                new_array_2 = np.array(new_array_2).reshape(self.IMG_SIZE, self.IMG_SIZE, 3)
                                new_array_2[:, :, 2][new_array_2[:, :, 2] <= 0] = 1.0 / 256.0
                                new_array_2 = np.divide(new_array_2[:, :, 0], new_array_2[:, :, 2])

                                bar = 0.5
                                new_array_2[new_array_2 < bar] = bar
                                new_array_2 = new_array_2 - bar
                                th = 1.2 * (1 - bar)
                                new_array_2[new_array_2 > th] = th
                                new_array_2 = new_array_2 / th
                                new_array_2[new_array_2 < 0.3] = 0.3

                                new_array_1 = np.array(new_array_1).reshape(self.IMG_SIZE, self.IMG_SIZE, 1)
                                new_array_2 = np.array(new_array_2).reshape(self.IMG_SIZE, self.IMG_SIZE, 1)

                        #data = [new_array_1, new_array_2, solar_zenith_angle, solar_azimuthal_angle, irradiance, file_name]

                        past_images.append([new_array_1, new_array_2])

        [img_short_lb0, img_long_lb0] = past_images[-1]
        [img_short_lb1, img_long_lb1] = past_images[-2]


        #totensor = ToTensor()
        sample = {'images': torch.from_numpy(np.concatenate((img_short_lb0, img_long_lb0, img_short_lb1, img_long_lb1))),
                  'aux_data': np.array(aux_data+past_irradiances),
                  'irradiance': np.array([target])}
        #sample = {'images': torch.from_numpy(new_array_1).float(), 'aux_data': np.array(aux_data),'irradiance': np.array([target])}

        #sample = totensor(sample)

        if self.transform:
            sample = self.transform(sample) #Error : variable transform / methode

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        images, aux_data, irradiance = sample['images'], sample['aux_data'], sample['irradiance']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        ##image = image.transpose((2, 0, 1))

        ###image = np.transpose(image, (2, 0, 1)).shape

        return {'images': image, #.type(torch.cuda.FloatTensor),
                'aux_data': torch.from_numpy(aux_data).float(),
                'irradiance': torch.from_numpy(irradiance).float()}

def show_image(image):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.pause(0.001)  # pause a bit so that plots are updated


def lookforward_index(y, m, d, h, minu, lookforward):  # return index of future sample
    id = neighbours(y, m, d, h, minu, lookforward)
    return (id)

def lookback_indexes(y, m, d, h, minu, lookback): #return list of past samples indexes
    list=[]
    for k in range(-lookback, 1):
        id=neighbours(y, m, d, h, minu, k)
        list.append(id)
    return(list)

def neighbours(y, m, d, h, minu, pos):
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


# Helper function to show a batch
def show_data_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, irradiance_batch = \
            sample_batched['image'], sample_batched['irradiance']
    batch_size = len(images_batch)

    for i in range(batch_size):
        plt.figure()
        plt.imshow(images_batch[i][:, :, 0])
        # plt.imshow(images_batch[i][:, :, 1])
        # plt.imshow(images_batch[i][:, :, 2])
        # plt.imshow(images_batch[i][:, :, 3])

        plt.title('Batch from dataloader (Irradiance : {})'.format(irradiance_batch[i]))
