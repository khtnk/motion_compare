# Motion Data Amplitude Analysis Tool - 実装ノート

## 概要
地震動データの振幅解析ツール `amplitude_analysis.py` の機能と使用方法についてまとめる。

## 主要機能

### メイン関数: `plot_max_amplitude_timeseries_with_waveforms`
指定キロ地点周辺の最大振幅を時系列でプロット（波形表示付き）

#### パラメータ
- `target_kilo`: 分析対象キロ地点
- `reference_path`: 基準データ（data0）のパス
- `dataset_info`: 分析対象データセット情報
- `kilo_range`: 全振幅抽出範囲（±、デフォルト: 0.005）
- `plot_kilo_start/end`: 波形表示範囲（Noneで自動設定）
- `analysis_kilo_start/end`: 分析用データの範囲
- `figsize`: 図のサイズ（デフォルト: (8, 16)）
- `show_original`: 元の波形も表示するかどうか（デフォルト: True）
- `enable_correction`: 波形位置補正を実行するかどうか（デフォルト: True） **[NEW]**

## コア機能

### 1. データ処理
- `load_and_resample_data`: CSVファイル読み込みとKiloベースリサンプリング
- `correct_kilo_by_correlation`: 相関最大化によるKilo軸補正（区間別）

### 2. 極値・振幅検出
- `simple_peak_detection`: 局所極大・極小検出と山谷交互フィルタリング
- `calculate_amplitude_analysis`: 隣接する極値間の全振幅計算

### 3. 時系列分析
- `get_max_amplitude_near_kilo`: 指定キロ範囲内の最大振幅抽出
- `extract_date_from_filepath`: ファイルパスから測定日時抽出（パターン: `NO\d+_(\d{8})\d{6}`）

### 4. 可視化
- `plot_waveforms_with_amplitude_range`: 複数波形の連続プロット（オフセット付き）
- 時系列グラフと波形グラフの統合表示

## 処理フロー

1. **データ読み込み**: 基準データ（data0）と分析対象データ（data1-12）を読み込み
2. **波形位置補正**: 
   - `enable_correction=True`: 基準データとの相関を最大化してKilo軸を補正
   - `enable_correction=False`: 元データをそのまま使用
3. **極値検出**: 各データセットで局所極大・極小を検出
4. **振幅計算**: 隣接する極値間の全振幅を計算
5. **時系列分析**: 指定キロ範囲内の最大振幅を抽出
6. **可視化**: 時系列グラフと波形プロットを生成

## 波形位置補正について

### 補正アルゴリズム
- 連続する3つの極値（P1, P2, P3）に注目
- P2のshift量を最適化して相関を最大化
- 区間[P1,P2]と[P2,P3]を線形スケーリングで調整
- 各区間で逐次的に補正を実行

### 補正の切り替え **[NEW]**
```python
# 補正有効（デフォルト）
result = plot_max_amplitude_timeseries_with_waveforms(
    target_kilo, reference_path, dataset_info
)

# 補正無効
result = plot_max_amplitude_timeseries_with_waveforms(
    target_kilo, reference_path, dataset_info, 
    enable_correction=False
)
```

## 特徴
- 基準データ（data0）に対して複数データセット（data1-12）を位置合わせ
- 日本語フォント対応（japanize_matplotlib）
- 元波形と補正後波形の同時表示可能
- 振幅統計情報の自動出力
- ファイルパスからの自動日付抽出

## 出力
- 時系列グラフ: 最大振幅の時系列変化
- 波形プロット: オフセット付き連続波形表示
- 統計情報: データセット数、振幅範囲、平均、標準偏差
- 振幅DataFrame: 詳細な解析結果

## 更新履歴
- 2024-09-04: 波形位置補正の有効/無効切り替え機能を追加（`enable_correction`パラメータ）