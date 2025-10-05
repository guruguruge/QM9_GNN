import torch
import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from typing import Tuple


def load_ncil_dataset() -> TUDataset:

    print("=" * 60)
    print("Step 1: データセットのロード")
    print("=" * 60)

    dataset_root = "./data/TUDataset"
    dataset = TUDataset(root=dataset_root, name="NCI1")

    print(f"Dataset: {dataset.name}")
    print(f"Number of Graphs: {len(dataset)}")
    print(f"Number od classes: {dataset.num_classes}")
    print(f"Node vector dimention: {dataset.num_node_features}")

    # サンプルグラフを確認
    print("\n--- Sample Graph ---")
    sample = dataset[0]
    print(f"sample.x.shape: {sample.x.shape}")  # [ノード数, 特徴次元]
    print(f"sample.edge_index.shape: {sample.edge_index.shape}")  # [2, エッジ数]
    print(f"sample.y: {sample.y}")  # グラフラベル
    print(f"Number of Nodes: {sample.num_nodes}")
    print(f"Number of edges: {sample.num_edges}")

    return dataset


def split_dataset(
    dataset: TUDataset, fold_idx: int = 0, n_splits: int = 10, random_state: int = 42
) -> Tuple[TUDataset, TUDataset]:

    print("\n" + "=" * 60)
    print(f"Step 2: データセットの分割 (Fold {fold_idx + 1}/{n_splits})")
    print("=" * 60)

    # StratifiedKFoldでクラス比率を保ったまま分割
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # ラベルを取得
    labels = np.array([data.y.item() for data in dataset])

    # fold_idxに対応する分割を取得
    for i, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        if i == fold_idx:
            break

    # データセットを分割
    train_dataset = dataset[train_idx.tolist()]
    val_dataset = dataset[val_idx.tolist()]

    print(f"訓練データ数: {len(train_dataset)}")
    print(f"検証データ数: {len(val_dataset)}")

    # クラス分布を確認
    train_labels = [data.y.item() for data in train_dataset]
    val_labels = [data.y.item() for data in val_dataset]
    print(f"訓練データのクラス分布: {np.bincount(train_labels)}")
    print(f"検証データのクラス分布: {np.bincount(val_labels)}")

    return train_dataset, val_dataset


def create_dataloaders(
    train_dataset: TUDataset, val_dataset: TUDataset, batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """
    DataLoaderを作成

    Args:
        train_dataset: 訓練用データセット
        val_dataset: 検証用データセット
        batch_size: バッチサイズ

    Returns:
        train_loader: 訓練用DataLoader
        val_loader: 検証用DataLoader
    """
    print("\n" + "=" * 60)
    print("Step 3: DataLoaderの作成")
    print("=" * 60)

    # DataLoaderを作成
    # shuffle=True: 訓練データはエポックごとにシャッフル
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows環境では0推奨
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(f"バッチサイズ: {batch_size}")
    print(f"訓練バッチ数: {len(train_loader)}")
    print(f"検証バッチ数: {len(val_loader)}")

    # サンプルバッチを確認
    print("\n--- サンプルバッチの構造 ---")
    sample_batch = next(iter(train_loader))
    print(f"batch.x.shape: {sample_batch.x.shape}")  # [全ノード数, 特徴次元]
    print(f"batch.edge_index.shape: {sample_batch.edge_index.shape}")  # [2, 全エッジ数]
    print(f"batch.y.shape: {sample_batch.y.shape}")  # [バッチサイズ]
    print(f"batch.batch.shape: {sample_batch.batch.shape}")  # [全ノード数]
    print(f"batch.batch: {sample_batch.batch}")  # どのノードがどのグラフか

    # バッチ内のグラフごとのノード数を確認
    print("\n--- バッチ内の各グラフのノード数 ---")
    unique, counts = torch.unique(sample_batch.batch, return_counts=True)
    for graph_id, node_count in zip(unique, counts):
        print(f"  グラフ{graph_id.item()}: {node_count.item()}ノード")

    return train_loader, val_loader
