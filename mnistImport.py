train_data = DataLoader(MNIST("/data", download=True, transform=ToTensor()),
                        batch_size=batch_size)
test_data = DataLoader(MNIST("/data", download=True,train=False, transform=ToTensor()),
                       batch_size=1000)