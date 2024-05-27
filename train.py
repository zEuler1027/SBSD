import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from model import QM9Dataset, pl_module
from config import train_config, model_config, data_config, utils_config
import os
import warnings
from pprint import pprint


# some configurations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128" # prevent OOM
job_id = os.environ.get('SLURM_JOB_ID', 'default_version') # get slurm_id
warnings.filterwarnings("ignore", message="An issue occurred while importing 'pyg-lib'.*") 
warnings.filterwarnings("ignore", message="An issue occurred while importing 'torch-sparse'.*") 

# use the same device as the data
device = data_config['device']

# create the dataset
dataset = QM9Dataset(**data_config)
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset,
    [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))],
    )

# create the dataloaders
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=train_config['batch_size'],
    collate_fn=dataset.collate_fn,
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=train_config['batch_size'],
    collate_fn=dataset.collate_fn,    
)

# print the config and data_size
config = train_config.copy()
config.update(model_config)
config.update(data_config)
config.update(utils_config)
pprint(config)
print(f"train_size: {len(train_dataset)}")
print(f"val_size: {len(val_dataset)}")

# create the lightning model
model = pl_module(train_config, model_config, utils_config)

# logger and callbacks
if not train_config['debugging']:
    logger = TensorBoardLogger("tb_logs", name=train_config['model_name'], version=job_id)
    checkpoint_callback = ModelCheckpoint(
    monitor="avg_val_loss",
    dirpath=f'tb_logs/{train_config["model_name"]}/{job_id}/checkpoints',
    filename="scorenet-{epoch:03d}-{avg_val_loss:.3f}",
    every_n_epochs=100,
    save_top_k=-1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=100)
    callbacks = [checkpoint_callback, progress_bar, lr_monitor]
else:
    logger = None
    callbacks = None

# devices and strategy
strategy = 'ddp_find_unused_parameters_true' if torch.cuda.is_available() else 'auto'
num_gpus = torch.cuda.device_count()

# create the trainer
trainer = L.Trainer(
    fast_dev_run=train_config['debugging'], 
    max_epochs=500,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    deterministic=False,
    logger=logger,
    devices=list(range(num_gpus)) if num_gpus > 1 else 1,
    strategy=strategy,
    log_every_n_steps=1,
    callbacks=callbacks,
    profiler=None,
    accumulate_grad_batches=1,
    limit_train_batches=None,
    limit_val_batches=200,
    # max_time="00:10:00:00",
)

# train the model
trainer.fit(model, train_loader, val_loader)
