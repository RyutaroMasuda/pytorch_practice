import torch

torch.set_printoptions(precision=4)

# 行：サンプル、列：特徴量
X_train = torch.tensor([
    [0.0,  0.0],
    [2.0, 20.0],
    [4.0, 40.0],
    [6.0, 60.0],
])

X_val = torch.tensor([
    [8.0,  80.0],
    [10.0, 100.0],
])

# TODO：この1行を修正する
stats_source = torch.cat([X_train, X_val], dim=0)

mu = stats_source.mean(dim=0)
sigma = stats_source.std(dim=0, correction=0).clamp_min(1e-8)

Z_train = (X_train - mu) / sigma
Z_val = (X_val - mu) / sigma

print("平均:", mu)
print("標準偏差:", sigma)
print("変換後の学習データの列平均:", Z_train.mean(dim=0))
print("変換後の検証データ:")
print(Z_val)