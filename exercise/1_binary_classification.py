import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42) # randnやlinear等の乱数の初期値を固定
torch.set_num_threads(1) # CPU上の計算に塚うスレッド数を1に設定

X = torch.randn(320,4) 
y = (X[:,0]-X[:,1]>0).float()

X_train, X_val = X[:256],X[256:] 
y_train, y_val = y[:256],y[256:]

loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=32,
    shuffle=True,
)

model = nn.Sequential(
    nn.Linear(4,16),
    nn.ReLU(),
    nn.Linear(16,1)
)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

for epoch in range(20):
    model.train()
    for xb, yb in loader:
        logits = model(xb).squeeze(-1)
        if epoch == 0:
            print("shapes:",logits.shape, yb.shape)

        loss = criterion(logits,yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
model.eval()
with torch.no_grad():
    val_logits = model(X_val).squeeze(-1)
    val_loss = criterion(val_logits, y_val)
    pred = (val_logits >= 0).float()
    accuracy = (pred == y_val).float().mean()

print(f"val_loss={val_loss.item():.4f}")
print(f"accuracy={accuracy.item():.3f}")
