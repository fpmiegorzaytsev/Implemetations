from flax import linen as nn


class DQNetworkDueling(nn.Module):
  n_actions: int

  @nn.compact
  def __call__(self, input): 
    output =  nn.Conv(features=32, kernel_size=(2, 2), strides=(1, 1))(input)
    output = nn.relu(output)
    output = nn.Conv(features=32, kernel_size=(2, 2), strides=(1, 1))(output)
    output = nn.relu(output)
    output = output.reshape((output.shape[0], -1))
    output = nn.Dense(256)(output)
    output = nn.relu(output)
    output = nn.Dense(128)(output)
    output = nn.relu(output)
    output = nn.Dense(self.n_actions)(output)
    return output


