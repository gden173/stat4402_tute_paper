class Encoder(nn.Module):
  def __init__(self, input_size, hidden_size1, hidden_size2):
    super(Encoder, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size1)    
    self.fc2 = nn.Linear(hidden_size1, hidden_size2) 
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.relu(self.fc1(x))   
    x = self.relu(self.fc2(x)) 
    return x