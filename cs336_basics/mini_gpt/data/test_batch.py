from data_loader import load_bin,get_batch

data = load_bin(
    "train.bin"
)

x,y = get_batch(
    data,
    batch_size=4,
    context_length=8,
    device="cpu",
)

print(x.shape)
print(y.shape)

print(x[0])
print(y[0])