import os
import json
import numpy as np


class RegMetrics:
    def __init__(self, mode, tensorboard, session_name, skill_score):
        self.mode = mode
        self.tensorboard = tensorboard
        self.session_name = session_name
        self.value = [0.0, 0.0]
        self.n_batch = 0
        self.skill_score = skill_score

        # minute by minute std
        #self.std_irradiance = 288.8
        # 15 min avg std
        self.std_irradiance = 254.4

    def update(self, pred, target):
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        #print('pred : ', pred)
        #print('target : ', target)

        #pred = np.argmax(pred, axis=-1)
        self.value[0] += ((pred - target)**2).mean()
        self.value[1] += (np.absolute(pred - target)).mean()
        self.n_batch += 1

    def evaluate(self, global_step):
        if self.n_batch > 0:
            scores = np.array(self.value) / self.n_batch
            #score = self.value / self.n_batch
        else:
            scores = [-float('inf'), -float('inf')]

        """if self.mode == 'train':
            fs_mse = 1 - scores[0]/self.skill_score.MSE_normalised_train
            fs_rmse = 1 - np.sqrt(scores[0] / self.skill_score.MSE_normalised_train)
            fs_mae = 1 - scores[1]/self.skill_score.MAE_normalised_train
        elif self.mode == 'val':
            fs_mse = 1 - scores[0] / self.skill_score.MSE_normalised_val
            fs_rmse = 1 - np.sqrt(scores[0] / self.skill_score.MSE_normalised_val)
            fs_mae = 1 - scores[1] / self.skill_score.MAE_normalised_val"""

        self.tensorboard.add_scalar(self.mode + '/mse', scores[0]*self.std_irradiance**2, global_step)
        #self.tensorboard.add_scalar(self.mode + '/fs_mse', fs_mse, global_step)
        self.tensorboard.add_scalar(self.mode + '/rmse', np.sqrt(scores[0]*self.std_irradiance**2), global_step)
        #self.tensorboard.add_scalar(self.mode + '/fs_rmse', fs_rmse, global_step)
        self.tensorboard.add_scalar(self.mode + '/mae', scores[1]*self.std_irradiance, global_step)

        #self.tensorboard.add_scalar(self.mode + '/fs_mae', fs_mae, global_step)

        self.save_json(scores)
        self.reset()
        return scores[0]

    def save_json(self, scores):
        filename = os.path.join(self.session_name, self.mode + '_metrics.json')
        output = {
            'MSE': scores[0]*self.std_irradiance**2,
            #'FS_MSE': 1 - scores[0]/self.skill_score.MSE_normalised_val,
            'rMSE': np.sqrt(scores[0]*self.std_irradiance**2),
            #'FS_rMSE': 1 - np.sqrt(scores[0]/self.skill_score.MSE_normalised_val),
            'MAE': scores[1]*self.std_irradiance,
            #'FS_MAE': 1 - scores[1]/self.skill_score.MAE_normalised_val
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

    def reset(self):
        self.value = [0.0, 0.0]
        self.n_batch = 0

