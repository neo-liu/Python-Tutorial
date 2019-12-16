# import tools.
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

# define ultra-parameters.
TIME_STEP = 50
INPUT_SIZE = 1
LR = 0.0002


# define RNN class and processions inside of the RNN class.
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        # redefine RNN class combined with ultra-parameters.
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )
        # construct fully-connected nerual network as usual to convert 2D signal into 1D signal.
        self.fc = nn.Linear(64, 1)

    def forward(self, x, h_state):
        # structure of output and hidden-state.
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)
        outs = [self.fc(r_out[:, time_step, :]) for time_step in range(r_out.size(1))]
        return torch.stack(outs, dim=1), h_state


rnn = RNN()
print(rnn)

# define optimizer and loss function.
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
criterion = nn.MSELoss()

h_state = None
# plot preparation .
plt.figure(1, figsize=(12, 5))
plt.ion() # keep on calculation.

# traning
for step in range(200):
    # construct data set.
    start, end = step * np.pi, (step+1) * np.pi
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32, endpoint=False)
    x_np = np.sin(steps)
    y_np = np.cos(steps)
    
    # keep in line with the same shape of data in 29 lines.
    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])
    
    # training.
    prediction, h_state = rnn(x, h_state)
    h_state = h_state.data

    loss = criterion(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # plot
    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw()
    plt.pause(0.02)

plt.ioff()
plt.show()
