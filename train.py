def train(model,data,model_name,do_every=10, num_epochs=600, lr=0.001):  
  model.to(device)
  model.train()
  optimizer = Adam(model.parameters(), lr=lr)
  for epoch in range(num_epochs):
    for batch_id, (x,_) in enumerate(data):
      model.zero_grad()
      x = x.reshape(-1, 28*28).to(device)
      y = (x > 0.5).float().to(device)
      z, (mu, log_variance), y_hat = model(x)
      loss = criterion(y_hat, y, mu, log_variance)
      loss.backward()
      optimizer.step()
    if (epoch % do_every ) == 0:
      torch.save(model.state_dict(), model_name)
      print("Epoch:{}, Loss:{}".format(epoch, loss.item()))
  model.cpu()
