import torch
import numpy as np

from dataload import *
from GIN_model import *
from train import train_one_epoch


def main():
    """
    メイン関数：データロードから訓練・評価まで
    """
    # ========================================
    # ハイパーパラメータ設定
    # ========================================
    config = {
        "input_dim": 37,
        "hidden_dim": 64,
        "num_layers": 5,
        "num_classes": 2,
        "batch_size": 32,
        "learning_rate": 0.01,
        "weight_decay": 5e-4,
        "num_epochs": 350,
        "fold_idx": 0,  # 使用するfold (0-9)
    }

    print("=" * 60)
    print("GIN on NCI1 Dataset")
    print("=" * 60)
    print("ハイパーパラメータ:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # ========================================
    # デバイス設定
    # ========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nデバイス: {device}")

    # ========================================
    # データセットのロードと分割
    # ========================================
    dataset = load_ncil_dataset()
    train_dataset, val_dataset = split_dataset(dataset, fold_idx=config["fold_idx"])
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, batch_size=config["batch_size"]
    )

    # ========================================
    # モデルの初期化
    # ========================================
    print("\n" + "=" * 60)
    print("Step 4: モデルの初期化")
    print("=" * 60)

    model = GIN(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_classes=config["num_classes"],
        epsilon=0.0,
        train_epsilon=False,
        pooling="sum",
    ).to(device)

    print(f"モデル: {model.__class__.__name__}")
    print(f"パラメータ数: {sum(p.numel() for p in model.parameters()):,}")

    # ========================================
    # オプティマイザとスケジューラの設定
    # ========================================
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # 学習率スケジューラ（50エポックごとに0.5倍）
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # ========================================
    # 訓練ループ
    # ========================================
    print("\n" + "=" * 60)
    print("Step 5: 訓練開始")
    print("=" * 60)

    best_val_acc = 0.0

    for epoch in range(1, config["num_epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        print("-" * 40)

        # 訓練
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)

        # 評価
        val_loss, val_acc = evaluate(model, val_loader, device)

        # 学習率更新
        scheduler.step()

        # 結果表示
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # ベストモデルの保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  ✓ Best model saved! (Val Acc: {best_val_acc:.4f})")

    # ========================================
    # 最終結果
    # ========================================
    print("\n" + "=" * 60)
    print("訓練完了")
    print("=" * 60)
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")


# ==============================
# 実行
# ==============================
if __name__ == "__main__":
    main()
