import json
import numpy as np
from datetime import datetime


def init_exp(exp_id):
    import os
    if not os.path.exists('saved_models/{}'.format(exp_id)):
        os.makedirs('saved_models/{}'.format(exp_id))
    if not os.path.exists('log/{}'.format(exp_id)):
        os.makedirs('log/{}'.format(exp_id))


def generate_probability_partition(num_entries):
    frac_vec = [0., 1.]
    for _ in range(num_entries-1):
        frac_vec.append(np.random.rand())
    frac_vec.sort()
    frac_vec = [frac_vec[j] - frac_vec[j - 1] for j in range(1, num_entries+1)]
    return frac_vec


class _Logger:
    @staticmethod
    def log_exp_info(exp_id, description):
        info_dict = {
            'exp_id': exp_id,
            'start_datetime': datetime.now().strftime("%m/%d/%Y %H:%M:%S"),
            'description': description,
        }
        with open('./log/{}/exp_info.json'.format(exp_id), 'w') as f:
            json.dump(info_dict, f, indent=4)

    @staticmethod
    def log_model_description(exp_id, model_fn):
        model = model_fn()
        with open('./log/{}/model_description.txt'.format(exp_id), 'w') as f:
            print(model, file=f)

    @staticmethod
    def log_server_solver(exp_id, server_solver):
        with open('./log/{}/server_solver.json'.format(exp_id), 'w') as f:
            json.dump(server_solver, f, indent=4)

    @staticmethod
    def log_client_solver(exp_id, client_solver):
        with open('./log/{}/client_solver.json'.format(exp_id), 'w') as f:
            json.dump(client_solver, f, indent=4)

    @staticmethod
    def log_client_preparation(exp_id, client_preparation_dict):
        with open('./log/{}/client_preparation.json'.format(exp_id), 'w') as f:
            json.dump(client_preparation_dict, f, indent=4)

    @staticmethod
    def log_validation_data(exp_id, validation_dict):
        with open('./log/{}/validation.json'.format(exp_id), 'w') as f:
            json.dump(validation_dict, f, indent=4)

    _client_selection_cache = {}
    @staticmethod
    def log_client_selection(exp_id, t, selection):
        _Logger._client_selection_cache[t] = selection
        with open('./log/{}/client_selection.json'.format(exp_id), 'w') as f:
            json.dump(_Logger._client_selection_cache, f, indent=4)


logger = _Logger()
