import torch


@torch.no_grad()
def retrieval_metrics(sim_matrix, ks=(1, 5, 10)):
    """
    sim_matrix: [N, N] où la diagonale correspond aux vraies paires
    """
    n = sim_matrix.size(0)
    device = sim_matrix.device
    targets = torch.arange(n, device=device)

    metrics = {}

    # image -> genes
    ranks_i2g = torch.argsort(sim_matrix, dim=1, descending=True)
    # genes -> image
    ranks_g2i = torch.argsort(sim_matrix.T, dim=1, descending=True)

    for k in ks:
        r_i2g = (ranks_i2g[:, :k] == targets.unsqueeze(1)).any(dim=1).float().mean().item()
        r_g2i = (ranks_g2i[:, :k] == targets.unsqueeze(1)).any(dim=1).float().mean().item()
        metrics[f"R@{k}_i2g"] = r_i2g
        metrics[f"R@{k}_g2i"] = r_g2i

    return metrics