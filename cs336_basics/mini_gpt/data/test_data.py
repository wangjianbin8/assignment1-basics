from data_loader import load_bin


train = load_bin(
    "train.bin"
)

print(train.shape)

print(train[:20])