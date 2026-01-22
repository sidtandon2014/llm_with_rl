Infrastructure: 8xL4 (24GB/ GPU)

Model: Qwen/Qwen3-0.6B 
# Calculations:

Total activations: 1627392
## Reference Model (one GPU):
Available memory = 20 GB
Model memory: 1.2 GB
S=512 (max generation length)
Activation Memory = 1627392 * B * S * 2 /(10^9) GB <= 19 GB

We can have following settings to accomodate everything in one GPU 
B=8 (Although we can increase this to 12 but for sequential procesisng we can have 8 similar to old policy model)

## Old Policy Model (one GPU) - This is required if we want to perform more than 1 batch updates 
Available memory = 20 GB
Model memory: 1.2 GB
S=512 (max generation length)

Activation Memory = 1627392 * B * S * 2 /(10^9) GB <= 19 GB

We can have following settings to accomodate everything in one GPU 
G=8
B=G=8

## Training Model (6 GPUs):

### Zero-0 setting (DDP)
Total available memory: 20GB
Optimizer: 3*4*N (Momentum + Variance + Parameters all in 32 bit) = 12*.6=7.2
Model: 2N = 1.2
Gradients: 2N = 1.2

Remaining: 10GB

GPUs = 6
S=512
Activation Memory = 1627392 * MBS * S * 2 /(10^9) GB = 1.67 * MBS <= 10 GB

MBS = 8 (Lets try if we can accomodate)
GBS = MBS x GPUs=8*6=48
GRADIENT accumulation = 1
