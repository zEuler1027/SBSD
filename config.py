import torch


train_config = dict(
    debugging = False,
    model_name = 'painn',
    batch_size = 32,
    lr = 1e-4,
)

model_config = dict(
    hidden_channels=256,
    out_channels=7,
    num_layers=6,
    num_rbf=256,
    cutoff=10.0,
)

data_config = dict(
    path = 'data/qm9/train.pkl', # path to the data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

utils_config = dict(
    diffschedule_sigma = 25.0,
)
