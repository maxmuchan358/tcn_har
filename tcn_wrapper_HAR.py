import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

from pytorch_tcn.tcn import TCN
from sklearn.metrics import confusion_matrix
import seaborn as sns

# データをロードする関数
def load_HAR_dataset():
    data_dir = "UCI HAR Dataset"
    X_train = np.loadtxt(os.path.join(data_dir, "train", "X_train.txt"))
    y_train = np.loadtxt(os.path.join(data_dir, "train", "y_train.txt")).astype(int)
    X_test = np.loadtxt(os.path.join(data_dir, "test", "X_test.txt"))
    y_test = np.loadtxt(os.path.join(data_dir, "test", "y_test.txt")).astype(int)
    return X_train, y_train, X_test, y_test

# データをロード
X_train, y_train, X_test, y_test = load_HAR_dataset()

# データの入力チャネル数を拡張
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# PyTorchのテンソルに変換
X_train = torch.Tensor(X_train).float()
y_train = torch.Tensor(y_train - 1).long()  # ラベルを0-indexedにする
X_test = torch.Tensor(X_test).float()
y_test = torch.Tensor(y_test - 1).long()  # ラベルを0-indexedにする

# データセットを作成
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# データローダーを作成（バッチサイズを100に設定）
batch_size = 100
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# モデルを定義
class TCNModel(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCNModel, self).__init__()
        self.tcn = TCN(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # TCNに入力を渡す
        y = self.tcn(x)
        # 最後のタイムステップの出力だけを使用して分類
        y = self.linear(y[:, :, -1])
        return y

# モデルを初期化
input_size = 561  # 入力の特徴量の数
output_size = 6  # クラスの数
num_channels = [64, 64, 64, 64]  # 各層のチャネル数
kernel_size = 3  # カーネルサイズ
dropout = 0.2  # ドロップアウト率

# モデルを初期化
model = TCNModel(input_size, output_size, num_channels, kernel_size, dropout)

# 損失関数と最適化手法を定義
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 学習の記録用
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# モデルをトレーニング
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()
    epoch_train_loss = running_loss / len(train_loader.dataset)
    epoch_train_accuracy = correct_train / total_train
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_accuracy)
    
    # バリデーションデータでの評価
    model.eval()
    running_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted_val = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted_val == labels).sum().item()
    epoch_val_loss = running_loss / len(test_loader.dataset)
    epoch_val_accuracy = correct_val / total_val
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.4f}, Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {epoch_val_accuracy:.4f}')

# 学習結果を保存
torch.save(model.state_dict(), 'tcn_har_model_batch100.pth')
np.savez('tcn_har_results_batch100.npz', train_losses=train_losses, val_losses=val_losses, train_accuracies=train_accuracies, val_accuracies=val_accuracies)

# 混同行列を作成
model.eval()
all_labels = []
all_predictions = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.numpy())
        all_predictions.extend(predicted.numpy())

cm = confusion_matrix(all_labels, all_predictions)

# 混同行列を可視化
plt.figure(figsize=(12, 8))

# 損失のグラフ
plt.subplot(2, 2, 1)
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Train and Validation Loss')

# 正解率のグラフ
plt.subplot(2, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy', color='blue')
plt.plot(val_accuracies, label='Validation Accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Train and Validation Accuracy')

# 混同行列を可視化
plt.subplot(2, 2, 3)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

# テストデータの予測結果を可視化
plt.subplot(2, 2, 4)
class_names = ['WALKING', 'WALKING UPSTAIRS', 'WALKING DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
plt.hist([y_test.numpy(), np.array(all_predictions)], bins=range(7), alpha=0.6, color=['orange', 'green'], label=['True Labels', 'Predicted Labels'])
plt.xticks(range(6), class_names, rotation=45)
plt.xlabel('Activity Label')
plt.ylabel('Count')
plt.legend()
plt.title('Test Predictions and True Labels')

plt.tight_layout()
plt.show()