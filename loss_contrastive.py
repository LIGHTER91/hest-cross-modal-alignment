import torch
import torch.nn.functional as F


def symmetric_info_nce(z_img, z_gene, temperature=0.07):
    logits = (z_img @ z_gene.T) / temperature
    targets = torch.arange(logits.size(0), device=logits.device)

    loss_i2g = F.cross_entropy(logits, targets)
    loss_g2i = F.cross_entropy(logits.T, targets)

    loss = 0.5 * (loss_i2g + loss_g2i)
    return loss, logits