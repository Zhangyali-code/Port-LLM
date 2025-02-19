from find_ports import *
import torch


def calculate_erro(target_ref, gen_target, target, criterion, device):
    # target_ref:(B,F,2)
    # gen_target:(B,F,2,100,50)
    # target:(B,F,2,100,50)
    B, F, _ = target_ref.shape

    # Create H_ref with broadcasting
    H_ref = target_ref.unsqueeze(-1).unsqueeze(-1).expand(B, F, 2, 100, 50).to(device)

    #erro_score: (B,F,2,100,50)
    erro_score = torch.abs(gen_target - H_ref)  # (B,F,2,100,50)

    # erro: (B,F,100,50)
    erro = torch.sqrt(erro_score[:, :, 0, :, :] ** 2 + erro_score[:, :, 1, :, :] ** 2)

    # min index (predicting the ports)
    index_x, index_y = optimize_indices(erro)

    # obtain the channel corresponding to the predicted port
    gen_href = torch.zeros(int(B), int(F), 2)
    for i in range(int(B)):
        for j in range(int(F)):
            gen_href[i, j, :] = target[i, j, :, int(index_x[i, j]), int(index_y[i, j])]
    gen_href = gen_href.to(device)

    # obtain the NMSE between the reference channel and the channel corresponding to the predicted port
    acc_loss = criterion(gen_href, target_ref) / criterion(target_ref, torch.zeros_like(target_ref))

    return index_x, index_y, acc_loss