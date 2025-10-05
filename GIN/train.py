import torch
import torch.nn.functional as F


def train_one_epoch(model, train_loader, optimizer, device):
    model.train()  # 訓練モード（Dropout, BatchNorm有効）

    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, data in enumerate(train_loader):
        # ========================================
        # データをデバイスに転送
        # ========================================
        data = data.to(device)
        # data.x: [全ノード数, 37] - バッチ内の全ノードの特徴
        # data.edge_index: [2, 全エッジ数] - バッチ内の全エッジ
        # data.y: [バッチサイズ] - 各グラフのラベル
        # data.batch: [全ノード数] - 各ノードがどのグラフに属するか

        # ========================================
        # 勾配をゼロにリセット
        # ========================================
        optimizer.zero_grad()

        # ========================================
        # 順伝播
        # ========================================
        # モデルにdataオブジェクトを渡す
        # 内部でdata.x, data.edge_index, data.batchが使われる
        out = model(data)  # [バッチサイズ, 2]

        # ========================================
        # 損失計算
        # ========================================
        loss = F.cross_entropy(out, data.y)

        # ========================================
        # 逆伝播
        # ========================================
        loss.backward()

        # ========================================
        # パラメータ更新
        # ========================================
        optimizer.step()

        # ========================================
        # メトリクスの計算
        # ========================================
        total_loss += loss.item() * data.num_graphs  # data.num_graphs = バッチサイズ
        pred = out.argmax(dim=1)  # 最大スコアのクラスを予測
        correct += (pred == data.y).sum().item()
        total += data.num_graphs

        # 進捗表示（10バッチごと）
        if (batch_idx + 1) % 10 == 0:
            print(
                f"  Batch {batch_idx + 1}/{len(train_loader)}: "
                f"Loss={loss.item():.4f}"
            )

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, val_loader, device):
    """
    検証データで評価
    """
    model.eval()  # 評価モード

    total_loss = 0
    correct = 0
    total = 0

    for data in val_loader:
        data = data.to(device)

        # 順伝播のみ
        out = model(data)
        loss = F.cross_entropy(out, data.y)

        # メトリクス計算
        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.num_graphs

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy
