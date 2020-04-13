from __future__ import print_function, division
import os
import cv2
import random
import numpy as np
from data.sirta.directories import data_images_dir
import socket

computer=socket.gethostname()


IMG_SIZE = 128
lookforward = 1
lookback = 1


def getitem(idx):
    DATADIR = data_images_dir(computer)

    y, m, d, h, minu = seq_indexes[idx]

    if m <= 9: M = '0{}'.format(m)
    else: M = '{}'.format(m)

    if d <= 9: D = '0{}'.format(d)
    else: D = '{}'.format(d)

    if h < 10: H = '0{}'.format(h)
    else: H = '{}'.format(h)

    if minu < 10: Minu = '0{}'.format(minu)
    else: Minu = '{}'.format(minu)

    folder_name = '{}{}{}'.format(y, M, D)
    path = os.path.join(DATADIR, folder_name)

    if os.path.isdir(path) == True:

        file_name = [y, m, d, h, minu]  # '2018{}{}{}{}'.format(M,D, H, minu)

        file_name_1 = '{}{}{}{}{}00_01.jpg'.format(y, M, D, H, Minu)
        file_name_2 = '{}{}{}{}{}00_03.jpg'.format(y, M, D, H, Minu)
        path_image_1 = os.path.join(path, file_name_1)
        path_image_2 = os.path.join(path, file_name_2)

        if os.path.isfile(path_image_1) == True and os.path.isfile(path_image_2) == True:
            img_array_1 = cv2.imread((path_image_1), cv2.IMREAD_GRAYSCALE)
            img_array_2 = cv2.imread((path_image_2), cv2.IMREAD_GRAYSCALE)

            img_array_redim_1 = img_array_1[35:725, 150:860]
            img_array_redim_2 = img_array_2[35:725, 150:860]

            new_array_1 = cv2.resize(img_array_redim_1, (IMG_SIZE, IMG_SIZE))
            new_array_2 = cv2.resize(img_array_redim_2, (IMG_SIZE, IMG_SIZE))

            new_array_1[new_array_1 == 0] = 1
            new_array_2[new_array_2 == 0] = 1

            img_short = np.array(new_array_1).reshape(1, IMG_SIZE, IMG_SIZE)
            img_long = np.array(new_array_2).reshape(1, IMG_SIZE, IMG_SIZE)

        else:
            print('os.path.isfile(path_image_1) : ', os.path.isfile(path_image_1))

    sample = {'short_exp': img_short, 'long_exp': img_long}

    return sample


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


def lookback_indexes(y, m, d, h, minu, lookback):  # return list of past samples indexes
    list = []
    for k in range(-lookback, 1):
        id = neighbours(y, m, d, h, minu, k)
        list.append(id)
    return (list)


def lookforward_index(y, m, d, h, minu, lookforward):  # return index of future sample
    id = neighbours(y, m, d, h, minu, lookforward)
    return (id)


def find_next_seq_index(y, m, d, h, minu, lookback, lookforward):  # return index of the next sequence
    # pos = lookback+lookforward+10+1
    pos = 2  # +lookback+lookforward+1

    id = neighbours(y, m, d, h, minu, pos)
    return (id)


def test_seq(y, m, d, h, minu, lookback, lookforward, computer):  # check if every sample of the sequence exists

    DATADIR = data_images_dir(computer)

    ans = True
    lb_id = lookback_indexes(y, m, d, h, minu, lookback)
    lf_id = lookforward_index(y, m, d, h, minu, lookforward)

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
            print('False path : ', path)
            ans = False
    return ans


def create_list(computer):  # create the list of sequence indexes

    DATADIR = data_images_dir(computer)

    print('\n>> Generation of the list of sequence indexes')
    list = []

    for Y in [2017, 2018]:

        for m in range(1, 13):  # range(1,13), m = month
            if m <= 9:
                M = '0{}'.format(m)
            else:
                M = '{}'.format(m)

            for d in range(2, 32):  # d:day
                if d <= 9:
                    D = '0{}'.format(d)
                else:
                    D = '{}'.format(d)
                folder_name = '{}/{}{}{}'.format(Y, Y, M, D)
                path = os.path.join(DATADIR, folder_name)

                print(path)

                if os.path.isdir(path) == True:
                    print(path)
                    h = 9
                    minu = random.randint(0, 29)
                    minu = int(2 * minu)
                    while h < 19:
                        if test_seq(Y, m, d, h, minu, lookback, lookforward, computer) == True:
                            list.append([y, m, d, h, minu])

                        Y, m, d, h, minu = find_next_seq_index(Y, m, d, h, minu, lookback, lookforward)


    print('\nNumber of Sequences available given the constraints :', len(list))

    return list

create_list(computer)