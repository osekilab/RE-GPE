"""
Improved batch samplers for garden-path experiments.

実装方針:
- 3つの言語現象（MVRR, NPZ, NPS）を効率的に学習
- ペア保持原則: 同一ペアは必ず同じbatch内に配置
- 現象バランス原則: 選択された現象内でバランス配分
"""

import logging
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class GardenPathBatchSampler(Sampler):
    """
    統一的なbatch sampler for garden-path experiments.

    現象選択パターンに応じて適切なバッチ構成を自動決定:
    - 単一現象(1/3): 特定現象への集中学習
    - 双現象(2/3): 2現象間の比較学習
    - 全現象(3/3): 全現象の汎化学習
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        selected_phenomena: Optional[List[str]] = None,
        shuffle: bool = True,
        drop_last: bool = False,
        epoch: int = 0,
    ):
        """
        Args:
            dataset: GardenPathDataset instance
            batch_size: バッチサイズ（偶数推奨）
            selected_phenomena: 学習対象の現象リスト（None=全現象）
            shuffle: エポック内でのペアシャッフル
            drop_last: 不完全なバッチを除外
            epoch: 現在のエポック番号（シード管理用）
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.selected_phenomena = self._normalize_phenomena(selected_phenomena)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = epoch

        # データ構造を解析
        self.pair_structure = self._analyze_dataset_structure()

        # バッチ構成戦略を決定
        self.strategy = self._determine_strategy()

        # バッチサイズの妥当性を検証
        self._validate_batch_size()

    def _normalize_phenomena(self, phenomena: Optional[List[str]]) -> Optional[List[str]]:
        """現象名を正規化（大文字化）"""
        if phenomena is None:
            return None
        return [p.upper() for p in phenomena]

    def _extract_metadata(self, idx: int) -> Tuple[str, str, str]:
        """データセットから必要なメタデータを抽出"""
        sample = self.dataset[idx]

        # Word-level data: 最初の非PAD値を取得
        construction = self._get_first_valid_value(sample.get("construction"))
        # pair_idフィールドを優先的に使用、なければitemフィールドを使用
        pair_id = self._get_first_valid_value(sample.get("pair_id"))
        if pair_id is None:
            pair_id = self._get_first_valid_value(sample.get("item"))
        ambiguity = self._get_first_valid_value(sample.get("ambiguity"))

        return (
            str(construction).upper() if construction else "UNKNOWN",
            str(pair_id) if pair_id else "UNKNOWN",
            str(ambiguity) if ambiguity else "UNKNOWN"
        )

    def _get_first_valid_value(self, data):
        """配列から最初の有効な値を取得"""
        if data is None:
            return None

        if isinstance(data, (torch.Tensor, np.ndarray)):
            data_flat = data.flatten()
            for val in data_flat:
                val_str = str(val)
                if val_str not in ["PAD", "", "-1"]:
                    return val
        else:
            if str(data) not in ["PAD", "", "-1"]:
                return data

        return None

    def _analyze_dataset_structure(self) -> Dict[str, Dict[str, List[int]]]:
        """
        データセット構造を解析して現象・ペア・文のインデックスを整理

        Returns:
            {phenomenon: {pair_id: [indices]}}
        """
        structure = {}

        for idx in range(len(self.dataset)):
            construction, pair_id, ambiguity = self._extract_metadata(idx)

            # 無効なデータをスキップ
            if construction == "UNKNOWN" or pair_id == "UNKNOWN":
                continue

            # 選択された現象のみを対象
            if self.selected_phenomena and construction not in self.selected_phenomena:
                continue

            if construction not in structure:
                structure[construction] = {}
            if pair_id not in structure[construction]:
                structure[construction][pair_id] = []

            structure[construction][pair_id].append(idx)

        # 統計情報をログ出力
        for phenomenon, pairs in structure.items():
            valid_pairs = sum(1 for indices in pairs.values() if len(indices) == 2)
            total_samples = sum(len(indices) for indices in pairs.values())
            logger.info(
                f"{phenomenon}: {len(pairs)} pairs, {valid_pairs} complete pairs, "
                f"{total_samples} total samples"
            )

        return structure

    def _determine_strategy(self) -> str:
        """現象数に基づいてバッチ構成戦略を決定"""
        num_phenomena = len(self.pair_structure)

        if num_phenomena == 0:
            raise ValueError("No valid data found for selected phenomena")
        elif num_phenomena == 1:
            return "single"  # 単一現象学習
        elif num_phenomena == 2:
            return "double"  # 双現象学習
        else:
            return "triple"  # 全現象学習

    def _validate_batch_size(self):
        """バッチサイズの妥当性を検証し、推奨値を提案"""
        num_phenomena = len(self.pair_structure)

        # 条件:
        # (i) ペアは必ず同じバッチ内に配置（バッチサイズは2の倍数）
        # (ii) 複数現象の場合、各バッチに全現象が含まれる

        if self.strategy == "single":
            # 単一現象: バッチサイズは2の倍数（ペア単位）
            if self.batch_size % 2 != 0:
                raise ValueError(
                    f"Batch size must be even for paired sampling. Got {self.batch_size}"
                )
            # 有効な値: 2, 4, 6, ..., 44
            valid_sizes = list(range(2, 45, 2))

        elif self.strategy == "double":
            # 双現象: 各現象から同数のペアを含む必要がある
            # バッチサイズは4の倍数（2現象×ペア×2文）
            if self.batch_size % 4 != 0:
                raise ValueError(
                    f"Batch size must be divisible by 4 for double phenomenon sampling "
                    f"(2 phenomena × 1 pair × 2 sentences minimum). Got {self.batch_size}"
                )
            # 有効な値: 4, 8, 12, ..., 88
            valid_sizes = list(range(4, 89, 4))

        else:  # triple
            # 全現象: 各現象から同数のペアを含む必要がある
            # バッチサイズは6の倍数（3現象×ペア×2文）
            if self.batch_size % 6 != 0:
                raise ValueError(
                    f"Batch size must be divisible by 6 for triple phenomenon sampling "
                    f"(3 phenomena × 1 pair × 2 sentences minimum). Got {self.batch_size}"
                )
            # 有効な値: 6, 12, 18, ..., 132
            valid_sizes = list(range(6, 133, 6))

        # 推奨値の表示
        if self.batch_size not in valid_sizes:
            closest = min(valid_sizes, key=lambda x: abs(x - self.batch_size))
            logger.warning(
                f"Batch size {self.batch_size} may not satisfy the pairing constraints. "
                f"Recommended sizes for {self.strategy} strategy: {valid_sizes[:10]}... "
                f"Closest valid size: {closest}"
            )

        # 各現象のペア数を確認
        pairs_per_phenomenon = {}
        for phenomenon, pairs in self.pair_structure.items():
            complete_pairs = sum(1 for indices in pairs.values() if len(indices) == 2)
            pairs_per_phenomenon[phenomenon] = complete_pairs

        # バッチあたりのペア数を計算
        if self.strategy == "single":
            pairs_per_batch = self.batch_size // 2
            max_pairs = pairs_per_phenomenon[list(self.pair_structure.keys())[0]]
        elif self.strategy == "double":
            pairs_per_batch = self.batch_size // 4  # 各現象から
            max_pairs = min(pairs_per_phenomenon.values())
        else:  # triple
            pairs_per_batch = self.batch_size // 6  # 各現象から
            max_pairs = min(pairs_per_phenomenon.values())

        logger.info(
            f"Batch configuration for {self.strategy} strategy:\n"
            f"  Batch size: {self.batch_size} sentences\n"
            f"  Pairs per phenomenon per batch: {pairs_per_batch}\n"
            f"  Available pairs per phenomenon: {pairs_per_phenomenon}\n"
            f"  Max complete batches: {max_pairs // pairs_per_batch}"
        )

    def _create_batches_single(self) -> List[List[int]]:
        """単一現象用のバッチ作成"""
        phenomenon = list(self.pair_structure.keys())[0]
        pairs = self.pair_structure[phenomenon]

        # ペアごとにインデックスを整理
        pair_indices = []
        for pair_id, indices in pairs.items():
            if len(indices) == 2:  # 完全なペアのみ
                pair_indices.append(indices)

        # シャッフル（エポックごとに異なるシード）
        if self.shuffle:
            rng = np.random.RandomState(self.epoch)
            rng.shuffle(pair_indices)

        # バッチ作成（ペア単位で配置）
        batches = []
        pairs_per_batch = self.batch_size // 2

        for i in range(0, len(pair_indices), pairs_per_batch):
            batch = []
            for j in range(i, min(i + pairs_per_batch, len(pair_indices))):
                batch.extend(pair_indices[j])

            if len(batch) >= self.batch_size or not self.drop_last:
                batches.append(batch)

        return batches

    def _create_batches_double(self) -> List[List[int]]:
        """双現象用のバッチ作成（各現象から均等配分）"""
        phenomena = list(self.pair_structure.keys())
        if len(phenomena) != 2:
            raise ValueError(f"Double strategy requires exactly 2 phenomena, got {len(phenomena)}")

        # 各現象のペアを整理
        all_pairs = {p: [] for p in phenomena}
        for phenomenon in phenomena:
            for pair_id, indices in self.pair_structure[phenomenon].items():
                if len(indices) == 2:
                    all_pairs[phenomenon].append(indices)

        # シャッフル
        if self.shuffle:
            rng = np.random.RandomState(self.epoch)
            for pairs in all_pairs.values():
                rng.shuffle(pairs)

        # バッチ作成（各現象から同数のペアを必ず含める）
        batches = []
        pairs_per_phenomenon = self.batch_size // 4  # 各現象からのペア数

        # 両現象の最小ペア数まで処理
        min_pairs = min(len(pairs) for pairs in all_pairs.values())

        for i in range(0, min_pairs, pairs_per_phenomenon):
            batch = []
            # 各現象から正確に同数のペアを取得
            for phenomenon in phenomena:
                for j in range(pairs_per_phenomenon):
                    if i + j < len(all_pairs[phenomenon]):
                        batch.extend(all_pairs[phenomenon][i + j])

            # バッチサイズが正確に一致するか、drop_lastがFalseの場合のみ追加
            if len(batch) == self.batch_size:
                batches.append(batch)
            elif not self.drop_last and len(batch) >= 4:  # 最低でも各現象1ペアは必要
                batches.append(batch)

        return batches

    def _create_batches_triple(self) -> List[List[int]]:
        """全現象用のバッチ作成（3現象から均等配分）"""
        phenomena = ["MVRR", "NPS", "NPZ"]  # 順序を固定

        # 実際に利用可能な現象を確認
        available_phenomena = [p for p in phenomena if p in self.pair_structure]
        if len(available_phenomena) != 3:
            raise ValueError(
                f"Triple strategy requires all 3 phenomena (MVRR, NPS, NPZ), "
                f"but only found: {available_phenomena}"
            )

        # 各現象のペアを整理
        all_pairs = {p: [] for p in phenomena}
        for phenomenon in phenomena:
            if phenomenon in self.pair_structure:
                for pair_id, indices in self.pair_structure[phenomenon].items():
                    if len(indices) == 2:
                        all_pairs[phenomenon].append(indices)

        # 各現象が最低1ペアは持っていることを確認
        for phenomenon, pairs in all_pairs.items():
            if len(pairs) == 0:
                raise ValueError(f"No complete pairs found for phenomenon {phenomenon}")

        # シャッフル
        if self.shuffle:
            rng = np.random.RandomState(self.epoch)
            for pairs in all_pairs.values():
                rng.shuffle(pairs)

        # バッチ作成（各現象から同数のペアを必ず含める）
        batches = []
        pairs_per_phenomenon = self.batch_size // 6  # 各現象からのペア数

        # 3現象の最小ペア数まで処理
        min_pairs = min(len(pairs) for pairs in all_pairs.values())

        for i in range(0, min_pairs, pairs_per_phenomenon):
            batch = []
            # 各現象から正確に同数のペアを取得
            for phenomenon in phenomena:
                for j in range(pairs_per_phenomenon):
                    if i + j < len(all_pairs[phenomenon]):
                        batch.extend(all_pairs[phenomenon][i + j])

            # バッチサイズが正確に一致するか、drop_lastがFalseの場合のみ追加
            if len(batch) == self.batch_size:
                batches.append(batch)
            elif not self.drop_last and len(batch) >= 6:  # 最低でも各現象1ペアは必要
                batches.append(batch)

        return batches

    def __iter__(self) -> Iterator[List[int]]:
        """バッチのイテレータを返す"""
        if self.strategy == "single":
            batches = self._create_batches_single()
        elif self.strategy == "double":
            batches = self._create_batches_double()
        else:  # triple
            batches = self._create_batches_triple()

        # バッチ全体のシャッフル（オプション）
        if self.shuffle:
            rng = np.random.RandomState(self.epoch + 1000)
            rng.shuffle(batches)

        return iter(batches)

    def __len__(self) -> int:
        """総バッチ数を返す"""
        # 簡易的な推定（実際のバッチ作成なしに計算）
        if self.strategy == "single":
            total_pairs = sum(
                1 for pairs in self.pair_structure.values()
                for indices in pairs.values()
                if len(indices) == 2
            )
            pairs_per_batch = self.batch_size // 2
            if self.drop_last:
                return total_pairs // pairs_per_batch
            else:
                return (total_pairs + pairs_per_batch - 1) // pairs_per_batch

        elif self.strategy == "double":
            min_pairs = min(
                sum(1 for indices in pairs.values() if len(indices) == 2)
                for pairs in self.pair_structure.values()
            )
            pairs_per_phenomenon = self.batch_size // 4
            if self.drop_last:
                return min_pairs // pairs_per_phenomenon
            else:
                return (min_pairs + pairs_per_phenomenon - 1) // pairs_per_phenomenon

        else:  # triple
            min_pairs = min(
                sum(1 for indices in pairs.values() if len(indices) == 2)
                for pairs in self.pair_structure.values()
            )
            pairs_per_phenomenon = self.batch_size // 6
            if self.drop_last:
                return min_pairs // pairs_per_phenomenon
            else:
                return (min_pairs + pairs_per_phenomenon - 1) // pairs_per_phenomenon

    def set_epoch(self, epoch: int):
        """エポックを設定（シャッフルのシード管理用）"""
        self.epoch = epoch

    def get_batch_info(self) -> Dict:
        """バッチ構成の詳細情報を返す（デバッグ用）"""
        return {
            "strategy": self.strategy,
            "batch_size": self.batch_size,
            "selected_phenomena": self.selected_phenomena,
            "num_phenomena": len(self.pair_structure),
            "phenomena_stats": {
                p: {
                    "num_pairs": len(pairs),
                    "complete_pairs": sum(1 for indices in pairs.values() if len(indices) == 2),
                    "total_samples": sum(len(indices) for indices in pairs.values())
                }
                for p, pairs in self.pair_structure.items()
            },
            "estimated_batches": len(self),
        }
