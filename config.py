import torch
from datetime import datetime


train_config = dict(
    debugging = False,
    epoch = 500,
    model_name = 'painn',
    batch_size = 128,
    lr = 1e-4,
    num_workers = 0,
    clip_grad = True,
)

model_config = dict(
    hidden_channels = 1024,
    out_channels = 7,
    num_layers = 8,
    num_rbf = 256,
    cutoff = 10.0,
)

data_config = dict(
    path = 'data/qm9/train.pkl', # path to the data
)

utils_config = dict(
    timepoint = datetime.now().strftime("%m-%d-%H:%M:%S"),
    sample_per_epoch = 50,
    diffschedule_sigma = 25.0,
    diffschedule_eps = 1e-5,
    sample_t_mode = 'linear',
    sample_steps = 1000,
)
