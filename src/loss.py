import torch
from torch import nn
import torch.nn.functional as F

class NT_xent(nn.Module):
    def __init__(self, temp=0.1) -> None:
        super(NT_xent, self).__init__()
        self.temp = temp

    def forward(self, mid_res):
        bert_moe_out, ent_moe_out, rel_ids = mid_res
        inter_prodcuts = bert_moe_out @ ent_moe_out.T 
        sent_norm = torch.linalg.norm(bert_moe_out, dim=-1)
        ent_norm = torch.linalg.norm(ent_moe_out, dim=-1) 
        mat_norm = torch.clamp(torch.outer(sent_norm, ent_norm), min=1e-8)
        cosine_sim = torch.exp((inter_prodcuts/mat_norm)/self.temp)
        cosine_sim_logit = torch.sum(cosine_sim, dim=0)
        pos_logit = torch.gather(cosine_sim, dim=0, index=torch.tensor([rel_ids], device=cosine_sim.device)).squeeze()
        nt_xent = torch.mean(-torch.log(pos_logit / cosine_sim_logit))

        return nt_xent
    
class NoisyLoss(nn.Module):
    def __init__(self, temp=0.1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.temp = temp

    def forward(self, noisy_input):
        output, output_noise = noisy_input # shape (N, hidden_dim)
        sim = F.cosine_similarity(output.unsqueeze(dim=1), output_noise, dim=-1)
        sim_tau = sim/self.temp

        loss = torch.exp(sim_tau.diag()) / torch.sum(torch.exp(sim_tau), dim=1)
        return torch.mean(-torch.log(loss))