class Latent(nn.Module):
  def __init__(self, hidden_size, latent_size):
    super(Latent, self).__init__()
    self.mu = nn.Linear(hidden_size, latent_size)
    self.log_variance = nn.Linear(hidden_size, latent_size)

  def forward(self, x):
    mu = self.mu(x)
    log_variance = self.log_variance(x)
    standard_deviation = torch.exp((1/2) * log_variance)
    epsilon = torch.randn_like(mu)
    return mu + standard_deviation * epsilon, (mu, log_variance)