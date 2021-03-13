def criterion(x,y, mu, log_variance):
  binary_cross_entropy = F.binary_cross_entropy(x,y, reduction="sum")
  kl_divergence = -(1/2) * torch.sum(1 + log_variance -mu.pow(2) - log_variance.exp())
  return binary_cross_entropy + kl_divergence