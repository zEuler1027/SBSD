import lightning as L
from model.en_diffusion import VESDE
from model.painn import PaiNN
from model.utils import DiffSchedule, ClipQueue, get_grad_norm
from model.io import write_batch_xyz
import torch
import os


class pl_module(L.LightningModule):
    def __init__(self, train_config, model_config, utils_config):
        super().__init__()
        self.save_hyperparameters()
        self.train_config = train_config
        self.utils_config = utils_config
        
        self.diffschedule = DiffSchedule(
            utils_config['diffschedule_sigma'],
            utils_config['diffschedule_eps'],
        )
        self.score_model = PaiNN(**model_config)
        self.en_diffusion = VESDE(
            self.score_model,
            self.diffschedule,
        )
        
        self.training_step_outputs = []
        self.validation_step_outputs = []
        if self.train_config['clip_grad']:
            self.gradnorm_queue = ClipQueue()
            self.gradnorm_queue.add(3000)

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
        self.log('train_loss', loss)
        return loss
    
    def on_train_epoch_end(self):
        outputs = self.training_step_outputs
        avg_loss = torch.stack(outputs).mean()
        self.log('avg_train_loss', avg_loss)
        print(f"Epoch {self.current_epoch}: avg_train_loss = {avg_loss}")
        self.training_step_outputs.clear()
        print('max_memory_allocated:{}GB'.format(torch.cuda.max_memory_allocated() / 1024**3 ))
    
    def validation_step(self, batch, batch_idx):
        # validation_step defines the train loop. It is independent of forward
        with torch.no_grad():
            if (self.current_epoch + 1) % self.utils_config['sample_per_epoch'] == 0 and batch_idx == 0:
                mask = batch['mask']
                atomic_numbers = batch['atomic_numbers']
                mols_pos = self.validate_sample(atomic_numbers, mask)
                time_point = self.utils_config['timepoint']
                save_dir = f'val_samples/{time_point}/epoch_{self.current_epoch}'
                os.makedirs(save_dir, exist_ok=True)
                write_batch_xyz(save_dir, atomic_numbers, mols_pos, mask)
                print(f"Epoch {self.current_epoch}: saved samples to {save_dir}")
            loss = self.compute_loss(batch)
            self.validation_step_outputs.append(loss)
            return loss
    
    def validate_sample(self, atomic_numbers, mask):
        mols_pos, _ = self.en_diffusion.sample(
            atomic_numbers,
            mask,
            num_steps=self.utils_config['sample_steps'],
            t_mode=self.utils_config['sample_t_mode'],
        )
        return mols_pos
    
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

    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val,
        gradient_clip_algorithm,    
    ):
        if not self.train_config['clip_grad']:
            return

        # Allow gradient norm to be 150% + 1.5 * stdev of the recent history.
        # max_grad_norm = 1.5 * self.gradnorm_queue.mean() + 3 * self.gradnorm_queue.std()
        max_grad_norm = 1.5 * self.gradnorm_queue.mean() + 3 * self.gradnorm_queue.std() # modified
        # Get current grad_norm
        params = [p for g in optimizer.param_groups for p in g["params"]]
        grad_norm = get_grad_norm(params)

        # Lightning will handle the gradient clipping
        self.clip_gradients(
            optimizer, 
            gradient_clip_val=max_grad_norm,
            gradient_clip_algorithm="norm",
        )

        if float(grad_norm) > max_grad_norm:
            self.gradnorm_queue.add(float(max_grad_norm))
        else:
            self.gradnorm_queue.add(float(grad_norm))

        if float(grad_norm) > max_grad_norm:
            print(
                f"Clipped gradient with value {grad_norm:.1f} "
                f"while allowed {max_grad_norm:.1f}"
            )
