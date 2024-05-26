import lightning as L
from model.en_diffusion import VESDE
from model.painn import PaiNN
from model.utils import DiffSchedule
import torch


class pl_module(L.LightningModule):
    def __init__(self, train_config, model_config, utils_config):
        super().__init__()
        self.save_hyperparameters()
        self.train_config = train_config
        self.utils_config = utils_config
        self.diffschedule = DiffSchedule(utils_config['diffschedule_sigma'])
        self.score_model = PaiNN(**model_config)
        self.en_diffusion = VESDE(
            self.score_model,
            self.diffschedule,
        )
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
    def compute_loss(self, batch):
        mask = batch['mask']
        atomic_numbers = batch['atomic_numbers']
        pos = batch['pos']
        l2loss = self.en_diffusion(pos, atomic_numbers, mask)
        return l2loss
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        loss = self.compute_loss(batch)
        self.training_step_outputs.append(loss)
        return loss
    
    def on_training_epoch_end(self):
        outputs = self.training_step_outputs
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('avg_train_loss', avg_loss)
        print(f"Epoch {self.current_epoch}: avg_train_loss = {avg_loss}")
        self.training_step_outputs.clear()
    
    def validation_step(self, batch, batch_idx):
        # validation_step defines the train loop. It is independent of forward
        loss = self.compute_loss(batch)
        self.log("val_loss", loss)
        self.validation_step_outputs.append(loss)
        return loss
    
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        avg_loss = torch.stack(outputs).mean()
        self.log('avg_val_loss', avg_loss)
        print(f"Epoch {self.current_epoch}: avg_val_loss = {avg_loss}")
        torch.cuda.empty_cache() 
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.en_diffusion.parameters(), lr=self.train_config['lr'])
        return optimizer
