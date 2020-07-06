import argparse
from trainers.trainer_sirta import SirtaTrainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model config/restore path.')
    parser.add_argument('--config', type=str, default='Experiments/sirta/experiments/sirta.yml', help='Path of the config file')
    #parser.add_argument('--config', type=str, default='', help='Path of the config file')
    parser.add_argument('--restore', type=str, default='', help='Path of the model to restore (weights, optimiser)')
    #parser.add_argument('--restore', type=str, default='Experiments/sirta/experiments/session_DESKTOP-A1O805T_2020_06_22_14_47_52_lstm_1hour_3imagesback', help='Path of the model to restore (weights, optimiser)')
    options = parser.parse_args()

    trainer = SirtaTrainer(options)
    trainer.train()

#python run_training.py --config Experiments/sirta/experiments/sirta.yml

# Windle
#ssh -A -Y -L 16006:127.0.0.1:6006 windle
#tensorboard --logdir=Documents/Cambridge/Research_project/deepcloud/Experiments/sirta --host localhost
