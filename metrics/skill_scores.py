import socket
import os
import pandas as pd
import numpy as np
import data.sirta.directories


class SkillScore():

    def __init__(self, shades, IMG_SIZE, lookback, lookforward, training_seq_indexes, validation_seq_indexes, step, helper):
        self.computer = socket.gethostname()
        self.shades = shades
        self.IMG_SIZE = IMG_SIZE
        self.lookback = lookback
        self.lookforward = lookforward
        self.training_seq_indexes = training_seq_indexes
        self.validation_seq_indexes = validation_seq_indexes
        self.step = step
        self.helper = helper

        self.MAE_train = 0
        self.MAE_normalised_train = 0
        self.MSE_train = 0
        self.MSE_normalised_train = 0
        self.rMSE_train = 0
        self.rMSE_normalised_train = 0
        self.MAE_val = 0
        self.MAE_normalised_val = 0
        self.MSE_val = 0
        self.MSE_normalised_val = 0
        self.rMSE_val = 0
        self.rMSE_normalised_val = 0

        # minute by minute std
        # self.std_irradiance = 288.8
        # 15 min avg std
        self.std_irradiance = 296.16
        self.smart_persistence_mean_error()


    def get_irradiance(self, y, m, d, h, minu):

        #m, d, h, minu = self.seq_indexes[idx]

        DATADIR = data.sirta.directories.data_sirta_grid(self.computer)
        DATADIR_IRRADIANCE = data.sirta.directories.data_irradiance_dir(self.computer)

        Y, M, D, H, Minu = self.helper.string_index(y, m, d, h, minu)

        folder_name = '{}'.format(Y)
        path = os.path.join(DATADIR, folder_name)

        if os.path.isdir(path) == True:

            irra_file_name = '{}/solys2_radflux_{}{}{}.csv'.format(Y, Y, M, D)
            irra_path = os.path.join(DATADIR_IRRADIANCE, irra_file_name)
            df = pd.read_csv(irra_path)

            min_1440 = minu + 60 * h
            irradiance_output = df['global_solar_flux'][min_1440]
            #solar_zenith_angle = df['solar_zenith_angle'][min_1440]
            #solar_azimuthal_angle = df['solar_azimuthal_angle'][min_1440]

        else:
            print('False path : ', path)

        return irradiance_output


    def absolute_error(self, target, forecast):
        AE = abs(target-forecast)
        return AE

    def squared_error(self, target, forecast):
        SE = (target-forecast)**2
        return SE

    def smart_persistence_forecast(self, Y, M, D, H, Minu, cs_data_irradiance, step):

        #self.get_irradiance(Y, M, D, H, Minu)
        current_irradiance = self.get_irradiance(Y, M, D, H, Minu)

        Minu1 = Minu - Minu%15
        if Minu1+15 == 60:
            Minu2 = Minu1
            Minu1 = Minu2-15
        else:
            Minu2 = Minu1+15

        CS_estimates = []

        for minu in [Minu1, Minu2]:
            indexes = [Y, M, D, H, minu]
            new_indexes = []

            for x in indexes:
                #print('x : ', x)
                if x<=9:
                    x='0{}'.format(x)
                elif x>9:
                    x='{}'.format(x)
                new_indexes.append(x)

            [y, m, d, h, min] = new_indexes
            date_id = '{}-{}-{}'.format(y, m, d)
            time_id = '{}:{}'.format(H, min)

            cs_data_irradiance_date = cs_data_irradiance[cs_data_irradiance['Date'] == date_id]
            cs_data_irradiance_time = cs_data_irradiance_date[cs_data_irradiance_date['Time'] == time_id]
            clr_sky_estimate = float(cs_data_irradiance_time['Clear-Sky'])

            CS_estimates.append(clr_sky_estimate)

        Irradiance_clr_sky_t1 = CS_estimates[0]
        Irradiance_clr_sky_t2 = CS_estimates[0] + (CS_estimates[1]-CS_estimates[0])*step/15

        if Irradiance_clr_sky_t1 ==0:
            prediction = current_irradiance
        else:
            prediction = current_irradiance*Irradiance_clr_sky_t2/Irradiance_clr_sky_t1

        return prediction


    def smart_persistence_mean_error(self): #layer_size, nb_training_seq, nb_epochs, update_frequency, learning_rate, decay_rate, batch_size,

        DATADIR_CLEAR_SKY_IRRADIANCE = data.sirta.directories.data_clear_sky_irradiance_dir(self.computer)
        file_path = '{}{}'.format(DATADIR_CLEAR_SKY_IRRADIANCE, 'SoDa_HC3-METEO_lat48.713_lon2.209_2017-01-01_2018-12-31_1266955311.csv')
        #file_path = '{}{}'.format(DATADIR_CLEAR_SKY_IRRADIANCE, 'SoDa_ClearSky_2018.csv')

        cs_data = pd.read_csv(file_path, engine='python', skiprows=34, header=0, parse_dates=True)
        cs_data_irradiance = cs_data[['Date', 'Time', 'Global Horiz', 'Clear-Sky']]
        #cs_data_irradiance = pd.read_csv(file_path, header=34, usecols=['Date', 'Time', 'Global Horiz', 'Clear-Sky'])


        #print('validation_seq_indexes  :', validation_seq_indexes)

        nb_samples = len(self.training_seq_indexes) + len(self.validation_seq_indexes)
        N = 0
        Set_ID = 0
        for set in [self.training_seq_indexes, self.validation_seq_indexes]:
            Set_ID += 1
            if Set_ID == 1:
                #print('\n >> Training Set :')
                training_ae_list = []
                training_se_list = []
            elif Set_ID == 2:
                #print('\n >> Validation Set :')
                validation_ae_list = []
                validation_se_list = []

            for seq_index in set:
                N+=1
                #print('seq_index : ', seq_index)
                if N%5000==0:
                    print('{}/{}'.format(N, nb_samples))

                #print(seq_index)
                y = 2018
                [m, d, h, minu] = seq_index
                #[y, m, d, h, minu] = seq_index
                lookback_seq_indexes = self.helper.lookback_indexes(y, m, d, h, minu)
                #print(lookback_seq_indexes)
                data_list = []
                lb_id = lookback_seq_indexes[-1]
                Y, M, D, H, Minu = lb_id

                step = self.lookforward*2 #1=2min
                forecast = self.smart_persistence_forecast(Y, M, D, H, Minu, cs_data_irradiance, step)

                # Target
                Y_tar, M_tar, D_tar, H_tar, Minu_tar = self.helper.lookforward_index(Y, M, D, H, Minu)
                target = self.get_irradiance(Y_tar, M_tar, D_tar, H_tar, Minu_tar)

                #print('target : ', target)
                #print('forecast : ', forecast)

                if Set_ID == 1:
                    training_ae_list.append(self.absolute_error(target, forecast))
                    #print('target, forecast : ', [target, forecast])
                    #print('squared_error(target, forecast) : ', squared_error(target, forecast))
                    training_se_list.append(self.squared_error(target, forecast))
                elif Set_ID == 2:
                    validation_ae_list.append(self.absolute_error(target, forecast))
                    validation_se_list.append(self.squared_error(target, forecast))


        self.MAE_train = np.mean(training_ae_list)
        self.MAE_normalised_train = self.MAE_train/self.std_irradiance
        self.MSE_train = np.mean(training_se_list)
        self.MSE_normalised_train = self.MSE_train/(self.std_irradiance**2)
        self.rMSE_train = np.sqrt(self.MSE_train)
        self.rMSE_normalised_train = np.sqrt(self.MSE_train)/self.std_irradiance

        self.MAE_val = np.mean(validation_ae_list)
        self.MAE_normalised_val = self.MAE_val/self.std_irradiance
        self.MSE_val = np.mean(validation_se_list)
        self.MSE_normalised_val = self.MSE_val/(self.std_irradiance**2)
        self.rMSE_val = np.sqrt(self.MSE_val)
        self.rMSE_normalised_val = np.sqrt(self.MSE_val)/self.std_irradiance

        print('\n Training set :')
        print('MAE = ', self.MAE_train)
        print('MAE (normalised) = ', self.MAE_normalised_train) #std_irradiance = 288.8
        print('MSE = ', self.MSE_train)
        print('MSE (normalised) = ', self.MSE_normalised_train)
        print('rMSE = ', self.rMSE_train)
        print('rMSE (normalised) = ', self.rMSE_normalised_train)

        print('\n Validation set :')
        print('MAE = ', self.MAE_val)
        print('MAE (normalised) = ', self.MAE_normalised_val)
        print('MSE = ', self.MSE_val)
        print('MSE (normalised) = ', self.MSE_normalised_val)
        print('rMSE = ', self.rMSE_val)
        print('rMSE (normalised) = ', self.rMSE_normalised_val)

        #training_errors = [MAE_train, MAE_normalised_train, MSE_train, MSE_normalised_train, rMSE_train, rMSE_normalised_train]
        #validation_errors = [MAE_val, MAE_normalised_val, MSE_val, MSE_normalised_val, rMSE_val, rMSE_normalised_val]

#computer = socket.gethostname()
#shades = 'Y'
#IMG_SIZE = 128 #256
#nb_training_seq = 1000#60203
#nb_epochs = 500
#update_frequency = 'batch'
#learning_rate = 1e-3
#decay_rate = 8e-6  # 1e-5
#batch_size = 32  # 11
#lookback = 1
#lookforward =1  # 1=2min
#layer_size = 8


#training_seq_indexes, val_seq_indexes = create_seq_index_list(computer, nb_training_seq, lookback, lookforward)

#print('Number of samples in the training set : ', len(training_seq_indexes))
#print('Number of samples in the validation set : ', len(val_seq_indexes))

#pickle_out = open(
    #        "datasets_X_y/fcst_{}_{}px_{}seq_{}ep_{}lb_{}lf.pickle".format(shades, IMG_SIZE, nb_training_seq, nb_epochs,
#                                                           lookback, lookforward), "wb")
#pickle.dump([training_seq_indexes, val_seq_indexes], pickle_out)
#pickle_out.close()


#errors = smart_persistence_mean_error(computer, shades, IMG_SIZE, layer_size, nb_training_seq, nb_epochs, update_frequency, learning_rate, decay_rate, batch_size, lookback, lookforward)
#print('Errors : ', errors)

