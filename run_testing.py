import argparse
from model_testing.model_test_sirta import SirtaTester
from data.sirta.SirtaDataset import SirtaDataset
import matplotlib.pyplot as plt
from utils import Helper
import numpy as np
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model config/restore path.')
    parser.add_argument('--config', type=str, default='', help='Path of the config file')
    parser.add_argument('--restore', type=str,
                        default='Experiments/sirta/experiments/'
                                'session_mario_2020_07_18_17_34_08_eumetsat_HRV_1hourForecast_highres_lr5e-4',
                        help='Path of the model to restore (weights, optimiser)')
    options = parser.parse_args()

    model_test = SirtaTester(options)
    helper = Helper(2, 2, 30)
    plot = True
    error = False
    start = time.time()
    if plot:
        time = []
        nowcast = []
        actual = []
        persistence = []
        MAE = 0
        MAEp = 0
        total = 0
        Y = 2018
        M = 8
        D = 6
        for H in range(10, 18):
            for i in range(0, 4):
                Minu = i * 15
                n, a, p = model_test.test([Y, M, D, H, Minu])
                # print('{}:{} = '.format(i, minu), n, a)
                _, _, _, H_lf, Minu_lf = helper.lookforward_index(Y, M, D, H, Minu)
                _, _, _, _, Minu_lf = helper.string_index(Y, M, D, H, Minu_lf)
                time.append('{}:{}'.format(H_lf, Minu_lf))
                nowcast.append(n)
                actual.append(a)
                persistence.append(p)
                MAE += np.abs(a - n)
                MAEp += np.abs(a - p)
                total += 1

        plt.plot(time, actual)
        plt.plot(time, nowcast)
        plt.plot(time, persistence, '-.')
        plt.title('{}/{}/{}'.format(D, M, Y))
        plt.title('{}/{}/{}  ||  MAE = {:0.2f} || MAE_per = {:0.2f}'.format(D, M, Y, MAE / total,
                                                                                       MAEp/total))
        plt.legend(['actual', 'nowcast', 'persistence'])
        plt.xticks(
            ['8:00', '9:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00',
             '17:00', '18:00'], rotation=90)
        plt.show()

    if error:
        SE = 0
        MSE_list = []
        AE = 0
        MAE_list = []
        month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        total = 0
        y = 2018
        for m in range(1, 13):
            SE = 0
            AE = 0
            total = 0
            start_month = time.time()
            for d in range(1, month[m-1] + 1):
                print(m, d)
                for h in range(5, 20):
                    for j in range(0, 4):
                        minu = j * 15
                        n, a = model_test.test([y, m, d, h, minu])
                        _, _, _, _, Minu = helper.string_index(y, m, d, h, minu)
                        er = (a - n) ** 2
                        SE += er
                        AE += np.sqrt(er)
                        total += 1

            end_month = time.time()
            print('Month {}: \nTotal SE: {:0.2f}, Total AE: {:0.2f}, Total Numbers: {}'.format(m, SE, AE, total))
            MSE = SE / total
            MAE = AE / total
            print('MSE: {:0.2f}, MAE: {:0.2f}'.format(MSE, MAE))
            print('{} min, {} seconds'.format(int((end_month - start_month) / 60), int((end_month - start_month) % 60)))
            MSE_list.append(MSE)
            MAE_list.append(MAE)

        end = time.time()
        print('Total Lists:')
        print('MSE:', MSE_list)
        print('MAE', MAE_list)
        print('{} min, {} seconds'.format(int((end-start) / 60), int((end - start) % 60)))

    #model_test.visualise_nowcast([2018, 7, 2, 12, 0])
    #model_test.visualise_nowcast([2018, 7, 2, 12, 15])
    #model_test.visualise_nowcast([2018, 7, 2, 12, 30])
    #model_test.visualise_nowcast([2018, 7, 2, 12, 45])
