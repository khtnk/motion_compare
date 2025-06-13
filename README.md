# Motion Data Amplitude Analysis Tool

指定キロ周辺の全振幅を時系列でプロット（波形表示付き）するツールです。

## ファイル構成

```
motion_compare/
├── amplitude_analysis.py        # メイン機能（すべての機能を統合）
├── data_matching_kilo.ipynb    # Jupyter Notebook版
└── README.md                   # このファイル
```

## メイン機能

### `plot_max_amplitude_timeseries_with_waveforms()`

指定したキロ地点周辺の全振幅を時系列で分析し、波形と一緒に可視化します。

**主な機能：**
- data0-data12の全データセットに対応
- 指定キロ周辺（±0.005m）の全振幅最大値を時系列でプロット
- data0-data12の波形を連続表示
- 全振幅抽出範囲を赤色でハイライト表示
- 波形表示範囲と全振幅抽出範囲を別々に設定可能

## 使用方法

### Pythonスクリプトから実行

```python
from amplitude_analysis import plot_max_amplitude_timeseries_with_waveforms

# 基本的な使用方法
result = plot_max_amplitude_timeseries_with_waveforms(
    target_kilo=283.25,      # 分析対象キロ
    kilo_range=0.005,        # 全振幅抽出範囲（±0.005m）
    plot_kilo_start=283.2,   # 波形表示開始キロ
    plot_kilo_end=283.3      # 波形表示終了キロ
)
```

### コマンドラインから実行

```bash
python amplitude_analysis.py
```

### Jupyter Notebookを使用

`data_matching_kilo.ipynb` を開いて対話的に実行

## パラメータ

- `target_kilo`: 分析対象キロ地点
- `kilo_range`: 全振幅抽出範囲（±）
- `plot_kilo_start/end`: 波形表示範囲（Noneで自動設定）
- `analysis_kilo_start/end`: 分析用データの範囲
- `figsize`: 図のサイズ
- `show_original`: 元の波形も表示するかどうか（デフォルト: True）

## 出力

1. **時系列グラフ**: 最大振幅の時間変化
2. **波形連続プロット**: data0-data12の波形を上下に表示
3. **統計情報**: 振幅の統計値（平均、標準偏差など）

## 必要なライブラリ

```bash
pip install pandas numpy matplotlib scipy
```

## データ形式

CSVファイルに以下の列が必要です：
- `Kilo`: 距離情報
- `UD`: 上下方向の振動データ