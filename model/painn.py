import math
from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter, segment_coo

from model.utils import ScaledSiLU, AtomEmbedding, RadialBasis, GaussianFourierProjection

class OAPaiNN(torch.nn.Module):
    def __init__(
            self,
            in_hidden_channels: int = 8,
            hidden_channels: int = 512,
            out_channels: int = 8,
            num_layers: int = 6,
            num_rbf: int = 128,
            cutoff: float = 10.0,
            rbf: Dict[str, str] = {'name': 'gaussian'},
            envelope: Dict[str, Union[str, float]] = {
                'name': 'polynomial',
                'exponent': 5,
            },
            num_elements: int = 83,
            **kwargs,
    ):
        super(OAPaiNN, self).__init__() 

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.cutoff = cutoff

        self.atom_emb = AtomEmbedding(hidden_channels, num_elements)
        self.embedding = nn.Linear(in_hidden_channels, hidden_channels)
        self.radial_basis = RadialBasis(
            num_radial=num_rbf,
            cutoff=self.cutoff,
            envelope=envelope,
            rbf=rbf,
        )
        self.global_radial_basis = RadialBasis(
            num_radial=num_rbf,
            cutoff=self.cutoff,
            envelope=envelope,
            rbf=rbf,
        )
        
        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()
        # added by sz, for global message
        self.global_message_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.global_message_layers.append(
                ScalarMessage(hidden_channels, num_rbf)
            )
            self.message_layers.append(
                PaiNNMessage(hidden_channels, num_rbf)
            )
            self.update_layers.append(
                PaiNNUpdate(hidden_channels)
            )
            
        self.out_xh = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            ScaledSiLU(),
            nn.Linear(hidden_channels // 2, out_channels),
        )

        self.out_dpos = PaiNNOutput(hidden_channels)

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.out_xh[0].weight)
        self.out_xh[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_xh[2].weight)
        self.out_xh[2].bias.data.fill_(0)

    def forward(self, xh, pos ,edge_index, **kwargs):
        subgraph_mask = kwargs.get('subgraph_mask', None).long()

        global_edge_index = edge_index.clone()
        edge_index = edge_index[:, subgraph_mask.squeeze().bool()]

        x = self.embedding(xh)
        vec = torch.zeros(x.size(0), 3, x.size(1)).to(x.device)

        diff = pos[edge_index[0]] - pos[edge_index[1]]
        dist = torch.norm(diff, p=2, dim=1)
        edge_rbf = self.radial_basis(dist)

        global_diff = pos[global_edge_index[0]] - pos[global_edge_index[1]]
        global_dist = torch.norm(global_diff, p=2, dim=1)
        global_edge_rbf = self.global_radial_basis(global_dist) * subgraph_mask
        
        for i in range(self.num_layers):
            s = self.global_message_layers[i](
                x,
                global_edge_index,
                global_edge_rbf,
            )
            x = x + s

            dx, dvec = self.message_layers[i](
                x,
                vec,
                edge_index,
                edge_rbf,
                diff,
            )
            x = x + dx
            vec = vec + dvec
            x = x * self.inv_sqrt_2

            dx, dvec = self.update_layers[i](x, vec)
            x = x + dx
            vec = vec + dvec
            x = x * self.inv_sqrt_2

        xh = self.out_xh(x).squeeze(1)
        dpos = self.out_dpos(x, vec)
        pos = pos + dpos

        return xh, pos, None
    


class PaiNN(torch.nn.Module):
    def __init__(
            self,
            hidden_channels: int = 512,
            out_channels: int = 8,
            num_layers: int = 6,
            num_rbf: int = 128,
            cutoff: float = 10.0,
            rbf: Dict[str, str] = {'name': 'gaussian'},
            envelope: Dict[str, Union[str, float]] = {
                'name': 'polynomial',
                'exponent': 5,
            },
            num_elements: int = 83,
            **kwargs,
    ):
        super(PaiNN, self).__init__() 

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.cutoff = cutoff

        self.atom_emb = AtomEmbedding(hidden_channels, num_elements)
        self.t_emb = GaussianFourierProjection(hidden_channels, num_rbf)
        self.embedding = nn.Linear(hidden_channels*2, hidden_channels)
        self.radial_basis = RadialBasis(
            num_radial=num_rbf,
            cutoff=self.cutoff,
            envelope=envelope,
            rbf=rbf,
        )
        
        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.message_layers.append(
                PaiNNMessage(hidden_channels, num_rbf)
            )
            self.update_layers.append(
                PaiNNUpdate(hidden_channels)
            )
            
        self.out_xh = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            ScaledSiLU(),
            nn.Linear(hidden_channels // 2, out_channels),
        )

        self.out_dpos = PaiNNOutput(hidden_channels)

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.out_xh[0].weight)
        self.out_xh[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_xh[2].weight)
        self.out_xh[2].bias.data.fill_(0)

    def forward(self, atomic_numbers, t, pos ,edge_index, **kargs):
        atom_emb = self.atom_emb(atomic_numbers)
        t_emb = self.t_emb(t)
        xh_t = torch.cat([atom_emb, t_emb], dim=-1)
        x = self.embedding(xh_t)
        vec = torch.zeros(x.size(0), 3, x.size(1)).to(x.device)

        diff = pos[edge_index[0]] - pos[edge_index[1]]
        dist = torch.norm(diff, p=2, dim=1)
        edge_rbf = self.radial_basis(dist)
        
        for i in range(self.num_layers):
            dx, dvec = self.message_layers[i](
                x,
                vec,
                edge_index,
                edge_rbf,
                diff,
            )
            x = x + dx + t_emb # add time embedding
            vec = vec + dvec
            x = x * self.inv_sqrt_2

            dx, dvec = self.update_layers[i](x, vec)
            x = x + dx
            vec = vec + dvec
            x = x * self.inv_sqrt_2

        dpos = self.out_dpos(x, vec)

        return dpos


class PaiNNMessage(MessagePassing):
    def __init__(
        self,
        hidden_channels,
        num_rbf,
    ) -> None:
        super(PaiNNMessage, self).__init__(aggr="add", node_dim=0)

        self.hidden_channels = hidden_channels

        self.x_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )
        self.rbf_proj = nn.Linear(num_rbf, hidden_channels * 3)

        self.inv_sqrt_3 = 1 / math.sqrt(3.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)
        self.x_layernorm = nn.LayerNorm(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.x_proj[0].weight)
        self.x_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.x_proj[2].weight)
        self.x_proj[2].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.rbf_proj.weight)
        self.rbf_proj.bias.data.fill_(0)
        self.x_layernorm.reset_parameters()

    def forward(
            self,
            x: torch.Tensor,
            vec: torch.Tensor, 
            edge_index: torch.Tensor, 
            edge_rbf:torch.Tensor, 
            edge_vector: torch.Tensor,
            ):
        
        xh = self.x_proj(self.x_layernorm(x))

        # TODO(@abhshkdz): Nans out with AMP here during backprop. Debug / fix.
        rbfh = self.rbf_proj(edge_rbf)

        # propagate_type: (xh: Tensor, vec: Tensor, rbfh_ij: Tensor, r_ij: Tensor)
        dx, dvec = self.propagate(
            edge_index,
            xh=xh,
            vec=vec,
            rbfh_ij=rbfh,
            r_ij=edge_vector,
            size=None,
        )

        return dx, dvec

    def message(self, xh_j, vec_j, rbfh_ij, r_ij):
        x, xh2, xh3 = torch.split(xh_j * rbfh_ij, self.hidden_channels, dim=-1)
        xh2 = xh2 * self.inv_sqrt_3

        vec = vec_j * xh2.unsqueeze(1) + xh3.unsqueeze(1) * r_ij.unsqueeze(2)
        vec = vec * self.inv_sqrt_h

        return x, vec

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec

    def update(
        self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs


class ScalarMessage(MessagePassing):
    def __init__(
        self,
        hidden_channels,
        num_rbf,
    ) -> None:
        super(ScalarMessage, self).__init__(aggr="mean", node_dim=0)

        self.hidden_channels = hidden_channels
        self.inv_sqrt_3 = 1 / math.sqrt(3.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)

        self.rbf_proj = nn.Linear(num_rbf, hidden_channels)

        self.x_layernorm = nn.LayerNorm(hidden_channels)

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels * 3),
            ScaledSiLU(),
            nn.Linear(hidden_channels * 3, hidden_channels * 3),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels * 2),
            ScaledSiLU(),
            nn.Linear(hidden_channels * 2, hidden_channels),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.rbf_proj.weight)
        self.rbf_proj.bias.data.fill_(0)
        self.x_layernorm.reset_parameters()

        nn.init.xavier_uniform_(self.edge_mlp[0].weight)
        self.edge_mlp[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.edge_mlp[2].weight)
        self.edge_mlp[2].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.node_mlp[0].weight)
        self.node_mlp[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.node_mlp[2].weight)
        self.node_mlp[2].bias.data.fill_(0)

    def forward(
            self, 
            x: torch.Tensor, 
            edge_index: torch.Tensor, 
            edge_rbf:torch.Tensor, 
            ):
        
        xh = self.x_layernorm(x)

        # TODO(@abhshkdz): Nans out with AMP here during backprop. Debug / fix.
        rbfh = self.rbf_proj(edge_rbf)

        # propagate_type: (xh: Tensor, vec: Tensor, rbfh_ij: Tensor, r_ij: Tensor)
        ds = self.propagate(
            edge_index,
            xh=xh,
            rbfh_ij=rbfh,
            size=None,
        )
        return ds

    def message(self, xh_i, xh_j, rbfh_ij):
        mij = torch.cat([xh_i, xh_j, rbfh_ij], dim=1)
        edge_weight = self.edge_mlp(mij)
        mij =  (mij + edge_weight) * self.inv_sqrt_3
        return mij

    def aggregate(
        self,
        features: torch.Tensor,
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = scatter(features, index, dim=self.node_dim, dim_size=dim_size, reduce='mean')
        return x

    def update(
        self, inputs: torch.Tensor,
    ) -> torch.Tensor:
        x = self.node_mlp(inputs)
        return x


class PaiNNUpdate(nn.Module):
    def __init__(self, hidden_channels) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels

        self.vec_proj = nn.Linear(
            hidden_channels, hidden_channels * 2, bias=False
        )
        self.xvec_proj = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.vec_proj.weight)
        nn.init.xavier_uniform_(self.xvec_proj[0].weight)
        self.xvec_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.xvec_proj[2].weight)
        self.xvec_proj[2].bias.data.fill_(0)

    def forward(self, x, vec):
        vec1, vec2 = torch.split(
            self.vec_proj(vec), self.hidden_channels, dim=-1
        )
        vec_dot = (vec1 * vec2).sum(dim=1) * self.inv_sqrt_h

        # NOTE: Can't use torch.norm because the gradient is NaN for input = 0.
        # Add an epsilon offset to make sure sqrt is always positive.
        x_vec_h = self.xvec_proj(
            torch.cat(
                [x, torch.sqrt(torch.sum(vec2**2, dim=-2) + 1e-8)], dim=-1
            )
        )
        xvec1, xvec2, xvec3 = torch.split(
            x_vec_h, self.hidden_channels, dim=-1
        )

        dx = xvec1 + xvec2 * vec_dot
        dx = dx * self.inv_sqrt_2

        dvec = xvec3.unsqueeze(1) * vec1

        return dx, dvec


class PaiNNOutput(nn.Module):
    def __init__(self, hidden_channels) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels

        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                ),
                GatedEquivariantBlock(hidden_channels // 2, 1),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.output_network:
            layer.reset_parameters()

    def forward(self, x, vec):
        for layer in self.output_network:
            x, vec = layer(x, vec)
        return vec.squeeze()


# Borrowed from TorchMD-Net
class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """

    def __init__(
        self,
        hidden_channels,
        out_channels,
    ) -> None:
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        self.vec1_proj = nn.Linear(
            hidden_channels, hidden_channels, bias=False
        )
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)

        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, out_channels * 2),
        )

        self.act = ScaledSiLU()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x, v):
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        x = self.act(x)
        return x, v

if __name__ == '__main__':
    model = PaiNN()
    atomic_numbers = torch.tensor([1, 6, 6, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    t = torch.tensor([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], dtype=torch.float)
    from torch_geometric.nn import radius_graph
    pos = torch.tensor([[ 0.0072, -0.5687,  0.0000],
        [-1.2854,  0.2499,  0.0000],
        [ 1.1304,  0.3147,  0.0000],
        [ 0.0392, -1.1972,  0.8900],
        [ 0.0392, -1.1972, -0.8900],
        [-1.3175,  0.8784,  0.8900],
        [-1.3175,  0.8784, -0.8900],
        [-2.1422, -0.4239,  0.0000],
        [ 1.9857, -0.1365,  0.0000]], dtype = torch.float)

    edge_index = radius_graph(pos, r=1.70, batch=None, loop=False)
    from e3nn import o3
    rot = o3.rand_matrix()
    pos_rot = pos @ rot
    out_rot = model(atomic_numbers, t, pos, edge_index) @ rot
    out = model(atomic_numbers, t, pos_rot, edge_index)
    print((out[1] - out_rot[1]).max())
    print((out[0] - out_rot[0]).max())
