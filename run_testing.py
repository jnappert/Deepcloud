import argparse
from model_testing.model_test_sirta import SirtaTester
from data.sirta.SirtaDataset import SirtaDataset
import matplotlib.pyplot as plt
from utils import Helper

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model config/restore path.')
    parser.add_argument('--config', type=str, default='', help='Path of the config file')
    parser.add_argument('--restore', type=str,
                        default='Experiments/sirta/experiments/session_DESKTOP-A1O805T_2020_06_02_23_46_45_nowcasting_min_by_min_5000images_months3+10',
                        help='Path of the model to restore (weights, optimiser)')
    options = parser.parse_args()

    model_test = SirtaTester(options)
    helper = Helper(0, 0, 15)
    nowcast = []
    actual = []
    time = []
    y = 2018
    m = 8
    d = 3
    for i in range(5, 20):
        for j in range(0, 4):
            minu = j * 15
            n, a = model_test.test([y, m, d, i, minu])
            # print('{}:{} = '.format(i, minu), n, a)
            _, _, _, _, Minu = helper.string_index(y, m, d, i, minu)
            time.append('{}:{}'.format(i, Minu))
            nowcast.append(n)
            actual.append(a)

    plt.plot(time, actual)
    plt.plot(time, nowcast)
    plt.title('{}/{}/{}'.format(d, m, y))
    plt.legend(['actual', 'nowcast'])
    plt.xticks(['5:00', '6:00', '7:00', '8:00', '9:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00',
                '17:00', '18:00', '19:00', '20:00'], rotation=90)
    #plt.ylim([0, 1000])
    plt.show()
    #model_test.visualise_nowcast([2018, 7, 2, 12, 0])
    #model_test.visualise_nowcast([2018, 7, 2, 12, 15])
    #model_test.visualise_nowcast([2018, 7, 2, 12, 30])
    #model_test.visualise_nowcast([2018, 7, 2, 12, 45])
