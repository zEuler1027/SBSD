import torch
from model.graph_tools import get_batch_mask_for_nodes, get_full_edges_index
from model.sample_tools import sample_center_gravity_zero_gaussian_batch
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
        edge_index = get_full_edges_index(nodes_mask, remove_self_edge=True)

        # sample zero CoM noise
        noise = sample_center_gravity_zero_gaussian_batch((pos.shape[0], pos_dim), nodes_mask)
        nodes_t = t[nodes_mask]
        std = self.schedule.marginal_prob_std(nodes_t)
        perturbed_pos = pos + noise * std[:, None]
        
        # compute score
        score = self.score_model(atomic_numbers, nodes_t, perturbed_pos, edge_index)
        l2loss = torch.mean(torch.sum((score * std[:, None] + noise)**2, dim=-1))
        return l2loss

    def sample(self, pos, atomic_numbers, mask):
        raise NotImplementedError

    
