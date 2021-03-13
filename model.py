 VAE(
  (encoder): Encoder(
    (fc1): Linear(in_features=784, out_features=512, bias=True)
    (fc2): Linear(in_features=512, out_features=32, bias=True)
    (relu): ReLU()
  )
  (latent): Latent(
    (mu): Linear(in_features=32, out_features=2, bias=True)
    (log_variance): Linear(in_features=32, out_features=2, bias=True)
  )
  (decoder): Decoder(
    (fc1): Linear(in_features=2, out_features=32, bias=True)
    (fc2): Linear(in_features=32, out_features=512, bias=True)
    (fc3): Linear(in_features=512, out_features=784, bias=True)
    (relu): ReLU()
    (sigmoid): Sigmoid()
  )
)