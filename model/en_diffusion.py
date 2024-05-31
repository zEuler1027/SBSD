import torch
from typing import Tuple, List
from model.graph_tools import get_batch_mask_for_nodes, get_full_edges_index
from model.sample_tools import sample_center_gravity_zero_gaussian_batch, remove_mean_batch
from model.utils import DiffSchedule


class VESDE(torch.nn.Module):
    def __init__(
        self,
        score_model: torch.nn.Module,
        schedule: DiffSchedule,
) -> None:
        super().__init__()
        self.score_model = score_model
        self.schedule = schedule
        
    def forward(self, pos, atomic_numbers, mask):
        t = self.schedule(mask)
        pos_dim = pos.shape[-1]
        nodes_mask = get_batch_mask_for_nodes(mask)
        edge_index = get_full_edges_index(nodes_mask, remove_self_edge=False)

        # sample zero CoM noise
        noise = sample_center_gravity_zero_gaussian_batch(
            (pos.shape[0], pos_dim), nodes_mask
        )
        nodes_t = t[nodes_mask]
        std = self.schedule.marginal_prob_std(nodes_t)
        perturbed_pos = pos + noise * std[:, None]
        
        # compute score
        score = self.score_model(atomic_numbers, nodes_t, perturbed_pos, edge_index)
        score = score / std[:, None] # normalize score
        if torch.any(torch.isnan(score)):
            print('nan in score, resetting to randn')
            score = torch.randn_like(score, requires_grad=True)
        score = remove_mean_batch(score, nodes_mask)
        
        l2loss = torch.mean(torch.sum((score * std[:, None] + noise)**2, dim=-1))
        return l2loss

    @torch.no_grad()
    def sample(
        self,
        atomic_numbers: torch.Tensor,
        mask: torch.Tensor,
        num_steps: int=500,
        t_mode: str='linear',
    )-> Tuple[torch.Tensor, List[torch.Tensor]]:
        '''
        Sample a mols and return the trajectory using Euler Maruyama sampler.
        '''
        device = atomic_numbers.device
        pos_shape = [atomic_numbers.size(0), 3]
        t = torch.ones(len(atomic_numbers), device=atomic_numbers.device)
        nodes_mask = get_batch_mask_for_nodes(mask)
        
        # sample zero CoM noise as initial position
        init_com = sample_center_gravity_zero_gaussian_batch(
            (pos_shape[0], pos_shape[1]), nodes_mask
        )
        init_pos = init_com * self.schedule.marginal_prob_std(t)[:, None]
        edge_index = get_full_edges_index(nodes_mask, remove_self_edge=False)
        num_steps = torch.tensor(num_steps, device=device)
        time_steps = self.schedule.sample_t(num_steps, mode=t_mode)
        step_sizes = torch.cat((-torch.diff(time_steps), time_steps[-1].unsqueeze(0)))
        
        # sample batch of mols
        pos = init_pos
        trajs = []
        for time_step, step_size in zip(time_steps, step_sizes):
            batch_time_step = torch.ones(pos.size(0), device=device) * time_step
            g = self.schedule.diffusion_coeff(batch_time_step)
            score = self.score_model(atomic_numbers, batch_time_step, pos, edge_index)
            
            # normalize score
            if torch.any(torch.isnan(score)):
                print('nan in score, resetting to randn')
                score = torch.randn_like(score)
            score = score / self.schedule.marginal_prob_std(batch_time_step)[:, None]
            score = remove_mean_batch(score, nodes_mask)
            mean_pos = pos + (g**2)[:, None] * score * step_size
            noise = sample_center_gravity_zero_gaussian_batch(
                (pos_shape[0], pos_shape[1]), nodes_mask
            )
            pos = mean_pos + torch.sqrt(step_size) * g[:, None] * noise
            trajs.append(mean_pos)
        return mean_pos, trajs
            
    @torch.no_grad()
    def pc_sample(self):
        raise NotImplementedError
    
    @torch.no_grad()
    def ode_smple(self):
        raise NotImplementedError
        
    
