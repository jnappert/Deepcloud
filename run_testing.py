import argparse
from model_testing.model_test_sirta import SirtaTester
from data.sirta.SirtaDataset import SirtaDataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model config/restore path.')
    parser.add_argument('--config', type=str, default='', help='Path of the config file')
    parser.add_argument('--restore', type=str, default='Experiments/sirta/experiments/session_DESKTOP-A1O805T_2020_05_21_15_48_27_testing_images_1000IMAGES', help='Path of the model to restore (weights, optimiser)')
    options = parser.parse_args()

    model_test = SirtaTester(options)
    model_test.test()
    #model_test.print_model()
    #model_test.forward_loss(batch, output)

#python run_training.py --config Experiments/sirta/experiments/sirta.yml

# Windle
#ssh -A -Y -L 16006:127.0.0.1:6006 windle
#tensorboard --logdir=Documents/Cambridge/Research_project/deepcloud/Experiments/sirta --host localhost
