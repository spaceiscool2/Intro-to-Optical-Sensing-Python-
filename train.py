import torch
import torch.nn as nn

from forward_process import (
        prepare_data,
        calculate_noise_xt
        )

class SimpleNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x, t, code):
        input_data = torch.hstack([x, t, code])
        return self.net(input_data)


def train(
    data,
    batch_size,
    device,
    epochs,
    diffusion_steps,
    alpha,
    learning_rate,
    output_model_path,
):
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    model = SimpleNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        count = 0
        epoch_loss = 0
        for x in data_loader:
            time_step = torch.randint(0, diffusion_steps, size=[len(x), 1]).to(device)
            image = x[:, :2]
            noise = x[:,2:4]
            code = x[:,4].unsqueeze(1)
            noised_x_t = calculate_noise_xt(image, alpha, noise, time_step)
            predicted_noise = model.forward(noised_x_t, time_step, code)
            loss = loss_fn(predicted_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            count += 1

        print("Epoch {0}, Loss={1}".format(epoch, round(epoch_loss / count, 5)))

    print("Finished training!!")
    torch.save(model.state_dict(), output_model_path)
    print("Saved model: ", output_model_path)


if __name__ == "__main__":
    torch.manual_seed(42)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Using {0} device".format(device))

    sample_num = 10000
    x1 = prepare_data(sample_num, code=1)
    x2 = prepare_data(sample_num, code=2)
    x = torch.vstack([x1, x2])

    data = torch.tensor(x, dtype=torch.float32).to(device)

    batch_size = 128
    epochs = 30
    diffusion_steps = 80
    alpha = 1. - 1.e-2
    learning_rate = 1e-3
    output_model_path = "trained_diffusion_model.pth"
    train(
        data,
        batch_size,
        device,
        epochs,
        diffusion_steps,
        alpha,
        learning_rate,
        output_model_path,
    )
