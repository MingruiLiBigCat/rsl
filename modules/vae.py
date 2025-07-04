import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.cnn1d import CNN1dEstimator

class VAE(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dims, 
                 hidden_dims, 
                 latent_dim = 16,
                 history_len = 20,
                 feet_contact_dim = 6,
                 kl_w = 0.1,
                 prior_mu=None):
        """
        Args:
            input_dim (int): Input dimension.
            hidden_dims (list of int): List of hidden layer dimensions.
            latent_dim (int): Dimension of the latent space.
        """
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.output_dims = output_dims

        self.encoder = CNN1dEstimator(nn.ReLU(), int(input_dim//history_len), history_len, hidden_dims[-1])

        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(hidden_dims[0], output_dims))
        self.decoder = nn.Sequential(*decoder_layers)
        # Feet contact decoder
        FC_decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            FC_decoder_layers.append(nn.Linear(prev_dim, h_dim))
            FC_decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        FC_decoder_layers.append(nn.Linear(hidden_dims[0], feet_contact_dim))
        FC_decoder_layers.append(nn.Sigmoid())
        self.FC_decoder = nn.Sequential(*FC_decoder_layers)
        self.prior_mu = prior_mu
        self.kl_w = kl_w


    def encode(self, x):
        """
        Encode input into latent distribution parameters.
        
        Args:
            x (tensor): Input tensor of shape (batch_size, input_dim)
        
        Returns:
            mu (tensor): Latent mean of shape (batch_size, latent_dim)
            logvar (tensor): Log variance of shape (batch_size, latent_dim)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for differentiable sampling.
        
        Args:
            mu (tensor): Latent mean of shape (batch_size, latent_dim)
            logvar (tensor): Log variance of shape (batch_size, latent_dim)
        
        Returns:
            z (tensor): Sampled latent vector of shape (batch_size, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def get_representation(self, x):
        """
        Get latent representation without reconstruction.
        
        Args:
            x (tensor): Input tensor of shape (batch_size, input_dim)
        
        Returns:
            z (tensor): Sampled latent vector of shape (batch_size, latent_dim)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z

    def decode(self, z):
        """
        Reconstruct outputs from latent vector.
        
        Args:
            z (tensor): Latent vector of shape (batch_size, latent_dim)
        
        Returns:
            recon_x (tensor): Main reconstruction of shape (batch_size, output_dims)
            recon_contact (tensor): Foot contact probabilities of shape (batch_size, feet_contact_dim)
        """
        return self.decoder(z), self.FC_decoder(z)

    def forward(self, x):
        """
        Full VAE forward pass.
        
        Args:
            x (tensor): Input tensor of shape (batch_size, input_dim)
        
        Returns:
            recon_x (tensor): Main reconstruction of shape (batch_size, output_dims)
            recon_contact (tensor): Foot contact probabilities of shape (batch_size, feet_contact_dim)
            mu (tensor): Latent mean of shape (batch_size, latent_dim)
            logvar (tensor): Log variance of shape (batch_size, latent_dim)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x, recon_contact= self.decode(z)
        return recon_x, recon_contact, mu, logvar