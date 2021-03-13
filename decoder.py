class Decoder(nn.Module):
  def __init__(self, latent_size, hidden_size1, hidden_size2,output_size):
    super(Decoder, self).__init__()
    self.fc1 = nn.Linear(latent_size, hidden_size1) 
    self.fc2 = nn.Linear(hidden_size1, hidden_size2)  
    self.fc3 = nn.Linear(hidden_size2, output_size)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):    
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.sigmoid(self.fc3(x))
    return x