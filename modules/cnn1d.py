
import torch.nn as nn

class CNN1dEstimator(nn.Module):
    """
    1D Convolutional Neural Network for processing sequential data.
    
    Attributes:
        activation_fn: Activation function used throughout the network
        tsteps: Fixed length of input sequences (temporal steps)
        encoder: Linear projection layer that expands feature dimensions
        conv_layers: 1D convolutional blocks configured for specific sequence lengths
        linear_output: Final processing layer producing the output
        
    Args:
        activation_fn: Activation function (e.g., nn.ReLU)
        input_size: Number of features per timestep
        tsteps: Number of timesteps in input sequences (must be 10, 20, or 50)
        output_size: Dimensionality of the output vector
        tanh_encoder_output: Unused in current implementation (retained for compatibility)
    """
    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
        """
        Initializes CNN1dEstimator with configurable architecture components.
        
        The network architecture dynamically adapts to different sequence lengths (tsteps):
        - For tsteps=50: Uses 3 convolutional layers with stride reduction
        - For tsteps=20: Uses 2 convolutional layers with stride reduction
        - For tsteps=10: Uses 2 convolutional layers with minimal reduction
        
        Raises:
            ValueError: If tsteps is not 10, 20, or 50
        """
        super(CNN1dEstimator, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps

        channel_size = 10

        self.encoder = nn.Sequential(
                nn.Linear(input_size, 3 * channel_size), self.activation_fn,
                )

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 8, stride = 4), self.activation_fn,
                    nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn,
                    nn.Conv1d(in_channels = channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn, nn.Flatten())
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 2, stride = 1), self.activation_fn,
                nn.Flatten())
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 6, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Flatten())
        else:
            raise(ValueError("tsteps must be 10, 20 or 50"))

        self.linear_output = nn.Sequential(
                nn.Linear(channel_size * 3, output_size), self.activation_fn
                )
        

    def forward(self, obs):
        """
        Forward pass processing:
        
        Input shape: (batch_size, tsteps, input_size)
        Output shape: (batch_size, output_size)
        
        Args:
            obs: Input tensor containing sequential data
        
        Returns:
            Processed output tensor
        """
        # nd * T * n_proprio
        nd = obs.shape[0]
        T = self.tsteps
        projection = self.encoder(obs.reshape([nd * T, -1])) # do projection for n_proprio -> 32
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        output = self.linear_output(output)
        return output