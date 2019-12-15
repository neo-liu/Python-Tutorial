import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

TIME_STEP = 20
INPUT_SIZE = 1
LearningRate = 0.0002


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)
        outs = [self.fc(r_out[:, step, :]) for step in range(r_out.size(1))]
        return torch.stack(outs, dim=1), h_state


rnn = RNN()
print(rnn)

optimizier = torch.optim.Adam(rnn.parameters(), lr=LearningRate)
criterion = nn.MSELoss()

plt.figure(1, figsize=(12, 5))
plt.ion()

h_state = None
for step in range(200):
    start, end = step*np.pi, (step+1)*np.pi
    steps = np.linspace(start, end, TIME_STEP)
    x_np = np.sin(step)
    y_np = np.cos(step)

    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

    prediction, h_state = rnn(x, h_state)
    h_state = h_state.data

    loss = criterion(prediction, y)
    optimizier.zero_grad()
    loss.backward()
    optimizier.step()

    plt.plot(steps, y_np.flatten(), 'r-', label='target(cos)')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-', label='prediction(sin->cos)')
    plt.draw()
    plt.pause(0.05)
    plt.legend(loc='best')

plt.ioff()
plt.show()

