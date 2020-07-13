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

from data.sirta.directories import data_images_dir, data_irradiance_dir, eumetsat_sat_images, \
    data_sirta_grid, data_clear_sky_irradiance_dir, processed_data_sirta_grid


# Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")

# plt.ion()   # interactive mode


class SirtaDataset(Dataset):
    """Sirta dataset."""

    def __init__(self, seq_indexes, shades, IMG_SIZE, lookback, lookforward, step, averaged_15min_dataset, mean, std,
                 helper, preprocessed_dataset=True, sat_images=True, transform=None):
        # def __init__(self, computer, nb_training_seq, lookback, lookforward, mode=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with measurement.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # self.measurements_list = pd.read_csv(csv_file)
        # self.root_dir = root_dir
        # self.nb_training_seq = nb_training_seq

        self.seq_indexes = seq_indexes
        self.shades = shades
        self.IMG_SIZE = IMG_SIZE
        self.lookback = lookback
        self.lookforward = lookforward
        self.computer = socket.gethostname()
        self.transform = transform
        self.preprocessed_dataset = preprocessed_dataset
        self.step = step
        self.averaged_15min_dataset = averaged_15min_dataset
        self.mean = mean
        self.std = std
        self.helper = helper
        self.sat_images = sat_images
        self.image_type = 'RGB_HRV'

        # self.training_seq_idexes = []
        # self.validation_seq_idexes = []
        # self.mode = mode

    def __len__(self):
        return len(self.seq_indexes)  # int(nb_training_seq*0.8) #len(self.measurements_list)

    def __getitem__(self, idx, train=True, lstm=False):

        #lstm = True

        if self.sat_images:
            DATADIR = eumetsat_sat_images(self.computer)
        elif self.preprocessed_dataset:
            DATADIR = processed_data_sirta_grid(self.computer)
        else:
            DATADIR = data_sirta_grid(self.computer)

        DATADIR_IRRADIANCE = data_irradiance_dir(self.computer)
        DATADIR_HELIO_IRRADIANCE = data_clear_sky_irradiance_dir(self.computer)

        helio = self.averaged_15min_dataset

        # y, m, d, h, minu = self.seq_indexes[idx]

        y = 2018
        if train:
            m, d, h, minu = self.seq_indexes[idx]
        else:
            y, m, d, h, minu = idx[0], idx[1], idx[2], idx[3], idx[4]

        samples_list_indexes = self.helper.lookback_indexes(y, m, d, h, minu)
        past_images = []
        past_irradiances = []
        aux_data = []

        for [y, m, d, h, minu] in samples_list_indexes:
            Y, M, D, H, Minu = self.helper.string_index(y, m, d, h, minu)

            folder_name = '{}'.format(Y)
            path = os.path.join(DATADIR, folder_name)

            if os.path.isdir(path) == True:

                irra_file_name = '{}/solys2_radflux_{}{}{}.csv'.format(Y, Y, M, D)
                irra_path = os.path.join(DATADIR_IRRADIANCE, irra_file_name)
                df = pd.read_csv(irra_path)
                min_1440 = minu + 60 * h

                if helio:
                    helio_path = '{}{}'.format(DATADIR_HELIO_IRRADIANCE,
                                               'SoDa_HC3-METEO_lat48.713_lon2.209_2017-01-01_2018-12-31_1266955311.csv')

                    helio_data = pd.read_csv(helio_path, header=34, usecols=['Date', 'Time', 'Global Horiz'])
                    date_id = '{}-{}-{}'.format(Y, M, D)
                    time_id = '{}:{}'.format(h, Minu)

                    helio_data_irradiance_date = helio_data[helio_data['Date'] == date_id]
                    helio_data_irradiance_time = helio_data_irradiance_date[
                        helio_data_irradiance_date['Time'] == time_id]
                    irradiance = float(helio_data_irradiance_time['Global Horiz']) * 4
                else:
                    irradiance = df['global_solar_flux'][min_1440]

                solar_zenith_angle = df['solar_zenith_angle'][min_1440]
                solar_azimuthal_angle = df['solar_azimuthal_angle'][min_1440]

                ### Zenith Angle
                solar_zenith_angle_rad = np.pi * solar_zenith_angle / 180
                solar_zenith_angle_cos = np.cos(solar_zenith_angle_rad)
                solar_zenith_angle_sin = np.sin(solar_zenith_angle_rad)

                ### Azimutal Angle
                solar_azimuthal_angle_rad = np.pi * solar_azimuthal_angle / 180
                solar_azimuthal_angle_cos = np.cos(solar_azimuthal_angle_rad)
                solar_azimuthal_angle_sin = np.sin(solar_azimuthal_angle_rad)

                file_name = [m, d, h, minu]  # '2018{}{}{}{}'.format(M,D, H, minu)

                # Target:
                y_target, m_target, d_target, h_target, minu_target = self.helper.lookforward_index(y, m, d, h, minu)
                Y_target, M_target, D_target, H_target, Minu_target = self.helper.string_index(y_target, m_target,
                                                                                               d_target, h_target,
                                                                                               minu_target)

                if helio:
                    target_date_id = '{}-{}-{}'.format(Y_target, M_target, D_target)
                    target_time_id = '{}:{}'.format(h_target, Minu_target)

                    helio_data_irradiance_date = helio_data[helio_data['Date'] == target_date_id]
                    helio_data_irradiance_time = helio_data_irradiance_date[
                        helio_data_irradiance_date['Time'] == target_time_id]
                    target = float(helio_data_irradiance_time['Global Horiz']) * 4
                else:
                    min_1440_target = minu_target + 60 * h_target
                    target = df['global_solar_flux'][min_1440_target]

                mean_irradiance = self.mean
                std_irradiance = self.std

                target = (target - mean_irradiance) / std_irradiance
                irradiance = (irradiance - mean_irradiance) / std_irradiance
                past_irradiances.append(irradiance)

                if lstm:
                    aux_data.append([solar_zenith_angle_rad, solar_zenith_angle_cos, solar_zenith_angle_sin,
                                     solar_azimuthal_angle_rad, solar_azimuthal_angle_cos, solar_azimuthal_angle_sin,
                                     irradiance])
                else:
                    aux_data = [solar_zenith_angle_rad, solar_zenith_angle_cos, solar_zenith_angle_sin,
                                solar_azimuthal_angle_rad, solar_azimuthal_angle_cos, solar_azimuthal_angle_sin]

                metadata_only = None
                if metadata_only == True:
                    aux_data = [solar_zenith_angle_rad, solar_zenith_angle_cos, solar_zenith_angle_sin,
                                solar_azimuthal_angle_rad, solar_azimuthal_angle_cos, solar_azimuthal_angle_sin,
                                irradiance]

                else:
                    if self.sat_images:
                        # Colour
                        file_name_1 = '{}{}/Colour/{}{}.jpg'.format(M, D, H, Minu)
                        # HRV
                        file_name_2 = '{}{}/HRV/{}{}.jpg'.format(M, D, H, Minu)
                    elif self.preprocessed_dataset:
                        file_name_1 = '{}{}/{}{}.jpg'.format(M, D, H, Minu)
                    else:
                        file_name_1 = 'Palaiseau_ghi_{}{}{}.csv'.format(y, M, D)
                    #file_name_2 = '{}{}{}{}{}00_03.jpg'.format(y, M, D, H, Minu)
                    path_image_1 = os.path.join(path, file_name_1)
                    path_image_2 = os.path.join(path, file_name_2)

                    if os.path.isfile(path_image_1) == True:

                        if self.shades == 'RGB' or self.shades == 'Bs' or self.shades == 'RBR':
                            img_array_BGR_1 = cv2.imread(path_image_1, 1)  # , cv2.IMREAD_GRAYSCALE)
                            img_array_1 = cv2.cvtColor(img_array_BGR_1, cv2.COLOR_BGR2RGB)
                            img_array_BGR_2 = cv2.imread(path_image_2, 1)  # , cv2.IMREAD_GRAYSCALE)
                            img_array_2 = cv2.cvtColor(img_array_BGR_2, cv2.COLOR_BGR2RGB)


                        elif self.shades == 'Y':
                            img_array_1 = cv2.imread((path_image_1), cv2.IMREAD_GRAYSCALE)
                            img_array_2 = cv2.imread((path_image_2), cv2.IMREAD_GRAYSCALE)

                        # -----------------------------------------------------------------------------------------
                        # image set up for satellite grid
                        # -----------------------------------------------------------------------------------------
                        elif self.shades == 'SAT':
                            if self.sat_images:
                                img_array_redim_1 = cv2.imread(path_image_1, 1)
                                img_array_redim_2 = cv2.imread(path_image_2, cv2.IMREAD_GRAYSCALE)
                                img_array_1 = cv2.cvtColor(img_array_redim_1, cv2.COLOR_BGR2RGB)
                                new_array_1 = cv2.resize(img_array_1, (self.IMG_SIZE, self.IMG_SIZE))
                                #new_array_1[new_array_1 == 0] = 1
                                new_array_1 = np.array(new_array_1).reshape(3, self.IMG_SIZE, self.IMG_SIZE)
                                new_array_2 = cv2.resize(img_array_redim_2, (self.IMG_SIZE, self.IMG_SIZE))
                                new_array_2 = np.array(new_array_2).reshape(1, self.IMG_SIZE, self.IMG_SIZE)
                            else:
                                if self.preprocessed_dataset:
                                    img_array_redim_1 = cv2.imread(path_image_1, cv2.IMREAD_GRAYSCALE)
                                else:
                                    img_array_redim_1 = np.reshape(get_sat_grid(y, m, d, h, minu), (41, 25))
                                new_array_1 = img_array_redim_1
                                #new_array_1 = cv2.resize(img_array_redim_1, (100, 164))
                                #new_array_1[new_array_1 == 0] = 1
                                new_array_1 = np.array(new_array_1).reshape(1, 41, 25)

                        if self.shades != 'SAT':
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

                        # data = [new_array_1, new_array_2, solar_zenith_angle, solar_azimuthal_angle, irradiance, file_name]

                        if self.shades != 'SAT':
                            past_images.append([new_array_1, new_array_2])
                        elif self.sat_images:
                            if self.image_type == 'RGB':
                                past_images.append(new_array_1)
                            if self.image_type == 'HRV':
                                past_images.append(new_array_2)
                            if self.image_type == 'RGB_HRV':
                                past_images = np.concatenate((new_array_1, new_array_2))
                        else:
                            past_images.append(new_array_1)

        if self.shades != 'SAT':
            [img_short_lb0, img_long_lb0] = past_images[-0]
            [img_short_lb1, img_long_lb1] = past_images[-1]

            # totensor = ToTensor()
            sample = {
                'images': torch.from_numpy(np.concatenate((img_short_lb0, img_long_lb0, img_short_lb1, img_long_lb1))),
                'aux_data': np.array(aux_data + past_irradiances),
                'irradiance': np.array([target])}
            # sample = {'images': torch.from_numpy(new_array_1).float(), 'aux_data': np.array(aux_data),'irradiance': np.array([target])}

            # sample = totensor(sample)
        else:
            # totensor = ToTensor()
            nowcast = True
            if not lstm and not nowcast:
                aux_data = aux_data + past_irradiances
            sample = {#'images': torch.from_numpy(np.concatenate((past_images[-0], past_images[-2], past_images[-1]))),  # forecasting
                      'images': torch.from_numpy(past_images), #[0]), # this is for regular just colour or hrv
                      'aux_data': np.array(aux_data),  # forecasting
                      'irradiance': np.array([target]),
                      'index': np.array(samples_list_indexes)}
            # sample = {'images': torch.from_numpy(new_array_1).float(), 'aux_data': np.array(aux_data),'irradiance': np.array([target])}

            # sample = totensor(sample)

        if self.transform:
            sample = self.transform(sample)  # Error : variable transform / methode

        return sample

    def get_image(self, idx):
        sample = self.__getitem__(idx, train=False)
        images, aux_data, irradiance, index = sample['images'], sample['aux_data'], sample['irradiance'], sample[
            'index']
        return {'images': images.unsqueeze(0),  # .type(torch.cuda.FloatTensor),
                'aux_data': torch.from_numpy(aux_data).float().unsqueeze(0),
                'irradiance': torch.from_numpy(irradiance).float().unsqueeze(0),
                'index': torch.from_numpy(index).float()}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        images, aux_data, irradiance, index = sample['images'], sample['aux_data'], sample['irradiance'], sample[
            'index']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        ##image = image.transpose((2, 0, 1))

        ###image = np.transpose(image, (2, 0, 1)).shape

        return {'images': image,  # .type(torch.cuda.FloatTensor),
                'aux_data': torch.from_numpy(aux_data).float(),
                'irradiance': torch.from_numpy(irradiance).float(),
                'index': torch.from_numpy(index).float()}


def show_image(image):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.pause(0.001)  # pause a bit so that plots are updated


def get_sat_grid(y, m, d, h, minu):
    computer = socket.gethostname()
    if d < 10:
        day = '0' + str(d)
    else:
        day = str(d)
    if m < 10:
        month = '0' + str(m)
    else:
        month = str(m)
    year = str(y)
    # round to nearest 15
    minu_rd = 15 * round(minu / 15)

    idx = int(h * 4 + minu_rd / 15 - 1)

    fp = data_sirta_grid(computer)
    fn = '{}/Palaiseau_ghi_{}{}{}.csv'.format(year, year, month, day)
    file = fp + fn

    # if os.path.isdir(file) == True:
    df = pd.read_csv(file, header=1, usecols=range(1, 1026))
    Irradiance = np.array(df.iloc[idx, :])

    return Irradiance / 100


# Helper function to show a batch
def show_data_batch(sample_batched, mean, std):
    """Show image with landmarks for a batch of samples."""
    images_batch, irradiance_batch, aux_data_batch, index_batch = \
        sample_batched['images'], sample_batched['irradiance'], sample_batched['aux_data'], sample_batched['index']

    for i in range(1, images_batch.size()[0]):
        #plt.figure(i*2 - 1)
        plt.figure(i)
        plt.imshow(images_batch[i, 0, :, :], cmap='gray', vmin=0, vmax=255)
        #plt.imshow(np.array(images_batch[i, 0:3, :, :]).reshape(156, 156, 3))
        # plt.imshow(images_batch[i, :, :], vmin=0, vmax=255)

        # plt.imshow(images_batch[i][:, :, 1])
        # plt.imshow(images_batch[i][:, :, 2])
        # plt.imshow(images_batch[i][:, :, 3])
        y = index_batch[i, 0, 0].item()
        m = index_batch[i, 0, 1].item()
        d = index_batch[i, 0, 2].item()
        h = index_batch[i, 0, 3].item()
        minu = index_batch[i, 0, 4].item()
        # plt.title('{}:{}, {}/{}/{}: Irradiance = {:0.2f}, Target = {:0.2f}'.format(str(h), str(minu), str(d), str(m), str(y),aux_data_batch[0][i + 6] * 288.8 + 434.4, irradiance_batch[0][0] * 288.8 + 434.4))
        plt.title(
            '{}:{}, {}/{}/{}: Irradiance = {:0.2f}'.format(str(h), str(minu), str(d), str(m), str(y),
                                                           irradiance_batch[i][0] * std + mean))
        #plt.figure(2*i)
        #plt.imshow(images_batch[i, 3, :, :], cmap='gray', vmin=0, vmax=255)
        plt.show()
        #plt.close()
