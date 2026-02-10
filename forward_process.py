import torch

def prepare_data(sample_num, code = 1):
    points = []
    if code == 1: # prepare data that give a triangle shape
        for _ in range(sample_num):
            side = torch.randint(0, 3, (1,)).item()
            if side == 0:
                x = torch.rand(1).item()
                y = 0.0
            elif side == 1:
                x = torch.rand(1).item() * 0.5
                y = x * (3 ** 0.5)
            else:
                x = 1 - torch.rand(1).item() * 0.5
                y = (1 - x) * (3 ** 0.5)

            points.append([x, y])
    elif code == 2: # prepare data that give a square shape
        for _ in range(sample_num):
            side = torch.randint(0, 4, (1,)).item()
            if side == 0:
                x = torch.rand(1).item()
                y = 0.0
            elif side == 1:
                x = 1.0
                y = torch.rand(1).item()
            elif side == 2:
                x = torch.rand(1).item()
                y = 1.0
            else:
                x = 0.0
                y = torch.rand(1).item()

            points.append([x, y])
    else:
        raise ValueError("code must be 1 (triangle) or 2 (square)")

    x = torch.tensor(points) + 0.01 * torch.randn(sample_num, 2)

    # Normalize value
    x = (x - x.mean()) / x.std()

    # augment x with noise and an embedding code
    embed_code = torch.full((sample_num, 1), code)
    x = torch.hstack([x, torch.randn_like(x), embed_code])
    return x

def calculate_noise_xt(x_0, alpha, noise, t):
    noised_x_t = pow(alpha, t) * x_0 + (1. - pow(alpha, t)) * noise
    return noised_x_t
