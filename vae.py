class VAE(nn.Module):
  def __init__(self, input_size, hidden_size1,hidden_size2, latent_size):    
    super(VAE, self).__init__()    
    self.encoder = Encoder(input_size,hidden_size1, hidden_size2)
    self.latent = Latent(hidden_size2, latent_size)    
    self.decoder = Decoder(latent_size, hidden_size2,
                           hidden_size1, input_size)
  
  def forward(self, x):
    x = self.encoder(x)
    z, (mu, log_variance) = self.latent(x)
    x = self.decoder(z)
    return z, (mu, log_variance), x 
 