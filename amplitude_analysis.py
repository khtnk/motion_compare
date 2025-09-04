"""
Motion Data Amplitude Analysis Tool - 修正版

メイン機能：plot_max_amplitude_timeseries_with_waveforms
指定キロ周辺の全振幅を時系列でプロット（波形表示付き）
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar
import re
from datetime import datetime
import japanize_matplotlib


def load_and_resample_data(file_path, kilo_interval=0.001):
    """CSVファイル読み込みとKiloベースリサンプリング"""
    df = pd.read_csv(os.path.join(file_path,'df_acc.csv'))
    
    # Kiloでソート（補間に必要）
    df = df.sort_values('Kilo').reset_index(drop=True)
    
    kilo_min = df['Kilo'].min()
    kilo_max = df['Kilo'].max()
    new_kilo = np.arange(kilo_min, kilo_max + kilo_interval, kilo_interval)
    
    # 補間実行
    resampled_ud = np.interp(new_kilo, df['Kilo'], df['UD'])
    
    return pd.DataFrame({
        'Kilo': new_kilo,
        'UD': resampled_ud
    })


def correct_kilo_by_correlation(data0, data1, kilo_start=None, kilo_end=None):
    """
    data0のUDを基準に、data1のKiloを補正する関数。
    極値間の相関を最大化して区間別にKilo軸を補正する。

    Args:
        data0 (pd.DataFrame): 基準となるデータ。'Kilo'と'UD'列を持つ。
        data1 (pd.DataFrame): 補正対象のデータ。'Kilo'と'UD'列を持つ。
        kilo_start: 補正対象範囲の開始
        kilo_end: 補正対象範囲の終了

    Returns:
        pd.DataFrame: Kiloが補正された新しいデータフレーム。
    """
    # 処理中のデータを変更するため、コピーを作成
    data1_corrected = data1.copy()

    # 分析範囲を設定
    if kilo_start is not None or kilo_end is not None:
        # 分析対象範囲のデータを抽出
        mask = pd.Series([True] * len(data1_corrected))
        if kilo_start is not None:
            mask &= data1_corrected['Kilo'] >= kilo_start
        if kilo_end is not None:
            mask &= data1_corrected['Kilo'] <= kilo_end
        
        analysis_data = data1_corrected[mask].copy()
        analysis_indices = data1_corrected[mask].index
        
        if len(analysis_data) < 10:
            print(f"分析範囲のデータが少なすぎます ({len(analysis_data)}点)。補正をスキップします。")
            return data1_corrected
    else:
        analysis_data = data1_corrected.copy()
        analysis_indices = data1_corrected.index

    # --- 1. 分析範囲のdata1から極値を検出 ---
    # simple_peak_detectionを使って極値を検出
    peaks = simple_peak_detection(analysis_data['UD'].values, analysis_data['Kilo'].values)
    
    if len(peaks) == 0:
        print("分析範囲で極値が検出されませんでした。")
        return data1_corrected
    
    # 極値のKilo値から元のデータフレームのインデックスを取得
    peak_indices = []
    for peak in peaks:
        # 最も近いKilo値のインデックスを見つける
        kilo_diff = np.abs(data1_corrected['Kilo'] - peak['kilo'])
        closest_idx = kilo_diff.idxmin()
        peak_indices.append(closest_idx)
    
    peak_indices = np.array(peak_indices)

    if len(peak_indices) < 3:
        print(f"分析範囲で検出された極値が3つ未満のため、処理をスキップします。({len(peak_indices)}個)")
        return data1_corrected

    print(f"分析範囲で検出された極値の数: {len(peak_indices)}")

    # --- 2. 逐次補正ループ（分析範囲内のみ） ---
    # 連続する3つの極値に注目してループ
    for i in range(len(peak_indices) - 2):
        # 現在のデータフレームから極値のインデックスを取得
        p1_idx, p2_idx, p3_idx = peak_indices[i], peak_indices[i+1], peak_indices[i+2]

        # 極値に対応するKilo値
        k_p1 = data1_corrected.loc[p1_idx, 'Kilo']
        k_p2 = data1_corrected.loc[p2_idx, 'Kilo']
        k_p3 = data1_corrected.loc[p3_idx, 'Kilo']

        # 最適化の目的関数を定義
        def objective_function(shift):
            """
            与えられたshift量での相関係数を計算する関数。
            最適化のために負の値を返す。
            """
            # --- 区間内のデータを伸縮させる ---
            # 区間 [p1, p2]
            mask_12 = (data1_corrected.index >= p1_idx) & (data1_corrected.index <= p2_idx)
            kilo_orig_12 = data1_corrected.loc[mask_12, 'Kilo']
            # 線形スケーリング
            kilo_new_12 = k_p1 + (kilo_orig_12 - k_p1) * (k_p2 + shift - k_p1) / (k_p2 - k_p1)

            # 区間 [p2, p3]
            mask_23 = (data1_corrected.index > p2_idx) & (data1_corrected.index <= p3_idx)
            kilo_orig_23 = data1_corrected.loc[mask_23, 'Kilo']
            # 線形スケーリング
            kilo_new_23 = (k_p2 + shift) + (kilo_orig_23 - k_p2) * (k_p3 - (k_p2 + shift)) / (k_p3 - k_p2)

            # ずらした後のKilo値とUD値
            kilo_shifted_section = np.concatenate([kilo_new_12.values, kilo_new_23.values])
            ud_section = data1_corrected.loc[mask_12 | mask_23, 'UD']

            # --- 相互相関の計算 ---
            # data0から対応するUD値を補間で取得
            # np.interpはソート済みのxpを要求するため、data0['Kilo']がソート済みであることを確認
            ud0_interp = np.interp(kilo_shifted_section, data0['Kilo'], data0['UD'])

            # 相関係数を計算
            correlation = np.corrcoef(ud_section, ud0_interp)[0, 1]

            # NaNの場合は相関なしとする
            if np.isnan(correlation):
                return 0
            
            # 最小化問題なので負の値を返す
            return -correlation

        # --- 3. 最適なshift量を探索 ---
        # p2がp1やp3を追い越さないように探索範囲を設定
        bounds = (-(k_p2 - k_p1) * 0.75, (k_p3 - k_p2) * 0.75) # マージンを持たせる

        # 最適化を実行
        result = minimize_scalar(objective_function, bounds=bounds, method='bounded')

        best_shift = result.x
        max_corr = -result.fun
        
        print(f"区間 {i+1} (Kilo: {k_p1:.2f}-{k_p3:.2f}): 最適なShift = {best_shift:.4f}, 最大相関 = {max_corr:.4f}")

        # --- 4. data1のKilo値を更新 ---
        # 最適なshiftを使って区間のKilo値を実際に更新する
        # 区間 [p1, p2]
        mask_12 = (data1_corrected.index >= p1_idx) & (data1_corrected.index <= p2_idx)
        kilo_orig_12 = data1_corrected.loc[mask_12, 'Kilo']
        kilo_new_12 = k_p1 + (kilo_orig_12 - k_p1) * (k_p2 + best_shift - k_p1) / (k_p2 - k_p1)
        data1_corrected.loc[mask_12, 'Kilo'] = kilo_new_12

        # 区間 [p2, p3]
        mask_23 = (data1_corrected.index > p2_idx) & (data1_corrected.index <= p3_idx)
        kilo_orig_23 = data1_corrected.loc[mask_23, 'Kilo']
        kilo_new_23 = (k_p2 + best_shift) + (kilo_orig_23 - k_p2) * (k_p3 - (k_p2 + best_shift)) / (k_p3 - k_p2)
        data1_corrected.loc[mask_23, 'Kilo'] = kilo_new_23

    return data1_corrected


def simple_peak_detection(data_values, kilo_values):
    """シンプルな極値検出"""
    peaks = []
    
    # 単純な局所極大・極小検出
    for i in range(1, len(data_values) - 1):
        if data_values[i] > data_values[i-1] and data_values[i] > data_values[i+1]:
            # 極大値
            peaks.append({
                'index': i,
                'kilo': kilo_values[i],
                'value': data_values[i],
                'type': 'max'
            })
        elif data_values[i] < data_values[i-1] and data_values[i] < data_values[i+1]:
            # 極小値
            peaks.append({
                'index': i,
                'kilo': kilo_values[i],
                'value': data_values[i],
                'type': 'min'
            })
    
    # キロ順にソート
    peaks.sort(key=lambda x: x['kilo'])
    
    # 山谷の交互フィルタリング
    if not peaks:
        return []
    
    filtered = [peaks[0]]
    for peak in peaks[1:]:
        if peak['type'] != filtered[-1]['type']:
            filtered.append(peak)
        else:
            # 同じ種類の場合、より極端な値を選択
            if peak['type'] == 'max' and peak['value'] > filtered[-1]['value']:
                filtered[-1] = peak
            elif peak['type'] == 'min' and peak['value'] < filtered[-1]['value']:
                filtered[-1] = peak
    
    return filtered


def calculate_amplitude_analysis(data, kilo_start=None, kilo_end=None):
    """全振幅を評価"""
    # 分析範囲を設定
    if kilo_start is not None or kilo_end is not None:
        mask = pd.Series([True] * len(data))
        if kilo_start is not None:
            mask &= data['Kilo'] >= kilo_start
        if kilo_end is not None:
            mask &= data['Kilo'] <= kilo_end
        analysis_data = data[mask].copy()
    else:
        analysis_data = data.copy()
    
    if len(analysis_data) < 10:
        return []
    
    # 極値検出
    peaks = simple_peak_detection(analysis_data['UD'].values, analysis_data['Kilo'].values)
    
    # 全振幅計算
    amplitude_results = []
    
    for i in range(len(peaks) - 1):
        current_peak = peaks[i]
        next_peak = peaks[i + 1]
        
        # 高低差の絶対値を計算
        amplitude = abs(next_peak['value'] - current_peak['value'])
        
        # キロの平均値（真ん中）
        avg_kilo = (current_peak['kilo'] + next_peak['kilo']) / 2
        
        amplitude_results.append({
            'kilo': avg_kilo,
            'amplitude': amplitude,
            'peak1_kilo': current_peak['kilo'],
            'peak1_value': current_peak['value'],
            'peak1_type': current_peak['type'],
            'peak2_kilo': next_peak['kilo'],
            'peak2_value': next_peak['value'],
            'peak2_type': next_peak['type']
        })
    
    return amplitude_results


def extract_date_from_filepath(filepath):
    """ファイルパスから取得日時を抽出"""
    pattern = r'NO\d+_(\d{8})\d{6}'
    match = re.search(pattern, filepath)
    
    if match:
        date_str = match.group(1)
        try:
            date_obj = datetime.strptime(date_str, '%Y%m%d')
            return date_obj.strftime('%Y-%m-%d')
        except ValueError:
            return None
    return None


def get_max_amplitude_near_kilo(amplitude_df, reference_path, dataset_info, target_kilo, kilo_range=0.005):
    
    dataset_info_add0 = dataset_info.copy()
    # data0のパスを追加
    dataset_info_add0['data0'] = reference_path
    
    kilo_min = target_kilo - kilo_range
    kilo_max = target_kilo + kilo_range
    
    results = []
    
    for dataset_name, path in dataset_info_add0.items():
        # 該当データセットの振幅データを抽出
        dataset_data = amplitude_df[amplitude_df['dataset'] == dataset_name]
        
        if len(dataset_data) == 0:
            continue
        
        # 指定キロ範囲内のデータを抽出
        range_data = dataset_data[
            (dataset_data['kilo'] >= kilo_min) & 
            (dataset_data['kilo'] <= kilo_max)
        ]
        
        # 日付を抽出
        date_str = extract_date_from_filepath(path)
        
        if len(range_data) > 0:
            max_amplitude = range_data['amplitude'].max()
            max_amplitude_kilo = range_data.loc[range_data['amplitude'].idxmax(), 'kilo']
            count = len(range_data)
        else:
            max_amplitude = 0.0
            max_amplitude_kilo = target_kilo
            count = 0
        
        results.append({
            'dataset': dataset_name,
            'date': date_str,
            'max_amplitude': max_amplitude,
            'max_amplitude_kilo': max_amplitude_kilo,
            'data_count': count,
            'target_kilo': target_kilo,
            'kilo_range': kilo_range
        })
    
    return pd.DataFrame(results)


def plot_waveforms_with_amplitude_range(reference_data, corrected_datasets, amplitude_df, dataset_info, reference_path,
                                       target_kilo, kilo_range=0.005,
                                       plot_kilo_start=None, plot_kilo_end=None,
                                       figsize=(12, 8), show_original=True):
    """波形の連続プロットと全振幅抽出範囲の可視化（1つのグラフに統合）"""
    if plot_kilo_start is None or plot_kilo_end is None:
        plot_kilo_min = target_kilo - 0.02
        plot_kilo_max = target_kilo + 0.02
    else:
        plot_kilo_min = plot_kilo_start
        plot_kilo_max = plot_kilo_end
    
    amp_kilo_min = target_kilo - kilo_range
    amp_kilo_max = target_kilo + kilo_range
    
    n_datasets = 1 + len(corrected_datasets)
    
    # 単一のグラフを作成
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # 13個の異なる色を確保するため、tab20カラーマップを使用
    if n_datasets <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
    else:
        # 13個以上の場合はtab20を使用
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_datasets]
    
    # data0プロット（オフセット無し）
    plot_mask = (reference_data['Kilo'] >= plot_kilo_min) & (reference_data['Kilo'] <= plot_kilo_max)
    plot_data = reference_data[plot_mask]
    
    # data0の測定日時を取得
    data0_date = extract_date_from_filepath(reference_path)
    data0_label = data0_date if data0_date else 'data0'
    
    if len(plot_data) > 0:
        ax.plot(plot_data['Kilo'], plot_data['UD'], color=colors[0], linewidth=2.0, alpha=0.8, label=data0_label)
        
        # 該当データセットの振幅ポイントをマーク
        if len(amplitude_df) > 0:
            dataset_amplitudes = amplitude_df[
                (amplitude_df['dataset'] == 'data0') & 
                (amplitude_df['kilo'] >= amp_kilo_min) & 
                (amplitude_df['kilo'] <= amp_kilo_max)
            ]
            
            for _, amp_row in dataset_amplitudes.iterrows():
                ax.scatter(amp_row['kilo'], 0, color='red', s=40, marker='v', alpha=0.8, zorder=5)
                ax.annotate(f'{amp_row["amplitude"]:.3f}', 
                           (amp_row['kilo'], 0),
                           textcoords="offset points", xytext=(0,15), ha='center',
                           fontsize=12, color='red', weight='bold')

    # data1-data12プロット（UDを-2ずつオフセット）
    for i, (dataset_name, data_dict) in enumerate(corrected_datasets.items()):
        corrected_data = data_dict['corrected']
        offset = -2 * (i + 1)  # -2, -4, -6, ..., -24
        
        # 測定日時を取得してラベルに使用
        dataset_date = extract_date_from_filepath(dataset_info.get(dataset_name, ''))
        dataset_label = dataset_date if dataset_date else dataset_name
        
        plot_mask = (corrected_data['Kilo'] >= plot_kilo_min) & (corrected_data['Kilo'] <= plot_kilo_max)
        plot_data = corrected_data[plot_mask]
        
        if len(plot_data) > 0:
            # 補正済み波形をプロット（オフセット適用）
            ax.plot(plot_data['Kilo'], plot_data['UD'] + offset, color=colors[i + 1], 
                   linewidth=2.0, alpha=0.8, label=dataset_label)
            
            # 元の波形も表示する場合（凡例なし）
            if show_original:
                original_data = data_dict['original']
                original_plot_mask = (original_data['Kilo'] >= plot_kilo_min) & (original_data['Kilo'] <= plot_kilo_max)
                original_plot_data = original_data[original_plot_mask]
                
                if len(original_plot_data) > 0:
                    ax.plot(original_plot_data['Kilo'], original_plot_data['UD'] + offset, 
                           color=colors[i + 1], linewidth=1.5, alpha=0.5, 
                           linestyle='--')
            
            # 該当データセットの振幅ポイントをマーク
            if len(amplitude_df) > 0:
                dataset_amplitudes = amplitude_df[
                    (amplitude_df['dataset'] == dataset_name) & 
                    (amplitude_df['kilo'] >= amp_kilo_min) & 
                    (amplitude_df['kilo'] <= amp_kilo_max)
                ]
                
                for _, amp_row in dataset_amplitudes.iterrows():
                    ax.scatter(amp_row['kilo'], offset, color='red', s=40, marker='v', alpha=0.8, zorder=5)
                    ax.annotate(f'{amp_row["amplitude"]:.3f}', 
                               (amp_row['kilo'], offset),
                               textcoords="offset points", xytext=(0,15), ha='center',
                               fontsize=12, color='red', weight='bold')
    
    # 全振幅抽出範囲とターゲットキロの表示
    ax.axvspan(amp_kilo_min, amp_kilo_max, alpha=0.15, color='red', 
              label=f'全振幅抽出範囲 (±{kilo_range:.3f}m)')
    ax.axvline(target_kilo, color='red', linestyle='--', alpha=0.7, 
              label=f'ターゲットキロ {target_kilo:.3f}')
    
    ax.set_xlabel('Kilo', fontsize=18)
    ax.set_ylabel('UD (オフセット適用)', fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    plt.tight_layout()
    plt.show()


def plot_max_amplitude_timeseries_with_waveforms(target_kilo, reference_path,
                                                 dataset_info, kilo_range=0.005,
                                                 plot_kilo_start=None, plot_kilo_end=None,
                                                 analysis_kilo_start=None, analysis_kilo_end=None,
                                                 figsize=(8, 16), show_original=True,
                                                 enable_correction=True):
    """
    メイン機能：指定キロ周辺の最大振幅を時系列でプロット（波形表示付き）
    
    Parameters:
    - target_kilo: 分析対象キロ地点
    - kilo_range: 全振幅抽出範囲（±）
    - plot_kilo_start/end: 波形表示範囲（Noneで自動設定）
    - analysis_kilo_start/end: 分析用データの範囲
    - figsize: 図のサイズ
    - show_original: 元の波形も表示するかどうか（デフォルト: True）
    - enable_correction: 波形位置補正を実行するかどうか（デフォルト: True）
    """
    
    print("=== 全振幅時系列分析（波形表示付き） ===")
    print(f"ターゲットキロ: {target_kilo}")
    print(f"全振幅抽出範囲: ±{kilo_range}m")
    print(f"波形位置補正: {'有効' if enable_correction else '無効'}")
    
    # データ読み込みと補正処理
    print("\\nデータ読み込み・補正処理中...")
    data0_up_res = load_and_resample_data(reference_path)
    
    # 補正処理
    corrected_datasets = {}
    for dataset_name, path in dataset_info.items():
        print(f"  {dataset_name}を処理中...")
        
        data = load_and_resample_data(path)
        original_data = data.copy()  # 補正前のデータを先に保存
        
        if enable_correction:
            corrected_data = correct_kilo_by_correlation(
                data0_up_res, data, analysis_kilo_start, analysis_kilo_end
            )
        else:
            corrected_data = data.copy()  # 補正しない場合は元データをそのまま使用
        
        corrected_datasets[dataset_name] = {
            'original': original_data,  # 元のデータを保持
            'corrected': corrected_data  # 補正後のデータ（補正なしの場合は元データと同じ）
        }
    
    # 全振幅分析
    print("\\n全振幅分析中...")
    all_amplitude_df = []
    
    # data0分析
    print(f"  data0を分析中...")
    data0_results = calculate_amplitude_analysis(data0_up_res, analysis_kilo_start, analysis_kilo_end)
    print(f"  data0で{len(data0_results)}個の振幅を検出")
    
    for result in data0_results:
        all_amplitude_df.append({
            'dataset': 'data0',
            'kilo': result['kilo'],
            'amplitude': result['amplitude'],
            'peak1_type': result['peak1_type'],
            'peak1_kilo': result['peak1_kilo'],
            'peak1_value': result['peak1_value'],
            'peak2_type': result['peak2_type'],
            'peak2_kilo': result['peak2_kilo'],
            'peak2_value': result['peak2_value']
        })
    
    # data1-data12分析
    for dataset_name, data_dict in corrected_datasets.items():
        corrected_data = data_dict['corrected']
        amplitude_results = calculate_amplitude_analysis(corrected_data, analysis_kilo_start, analysis_kilo_end)
        print(f"  {dataset_name}で{len(amplitude_results)}個の振幅を検出")
        
        for result in amplitude_results:
            all_amplitude_df.append({
                'dataset': dataset_name,
                'kilo': result['kilo'],
                'amplitude': result['amplitude'],
                'peak1_type': result['peak1_type'],
                'peak1_kilo': result['peak1_kilo'],
                'peak1_value': result['peak1_value'],
                'peak2_type': result['peak2_type'],
                'peak2_kilo': result['peak2_kilo'],
                'peak2_value': result['peak2_value']
            })
    
    if not all_amplitude_df:
        print("振幅データが見つかりませんでした。")
        return None
        
    amplitude_df = pd.DataFrame(all_amplitude_df)
    print(f"振幅データ {len(amplitude_df)} 件を作成しました。")
    
    # 時系列分析
    max_amp_df = get_max_amplitude_near_kilo(amplitude_df, reference_path, dataset_info, target_kilo, kilo_range)
    max_amp_df = max_amp_df.dropna(subset=['date']).copy()
    max_amp_df['date_obj'] = pd.to_datetime(max_amp_df['date'])
    max_amp_df = max_amp_df.sort_values('date_obj')
    
    if len(max_amp_df) == 0:
        print("プロット可能なデータがありません")
        return None
    
    # 時系列プロット
    fig, ax1 = plt.subplots(1, 1, figsize=(12,4), sharex=True)
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,8), sharex=True)
    
    ax1.plot(max_amp_df['date_obj'], max_amp_df['max_amplitude'], 
             marker='o', linewidth=2, markersize=8, color='blue')
    ax1.set_ylabel('最大振幅')
    ax1.set_title(f'キロ {target_kilo:.3f}±{kilo_range:.3f}m 周辺の最大振幅時系列変化')
    ax1.grid(True, alpha=0.3)
    
    for i, row in max_amp_df.iterrows():
        ax1.annotate(row['dataset'], 
                    (row['date_obj'], row['max_amplitude']),
                    textcoords="offset points", xytext=(0,10), ha='center',
                    fontsize=8, alpha=0.7)
    
    # ax2.bar(max_amp_df['date_obj'], max_amp_df['data_count'], 
    #         alpha=0.7, color='orange', width=10)
    # ax2.set_ylabel('データ点数')
    # ax2.set_xlabel('取得日')
    # ax2.set_title('範囲内データ点数')
    # ax2.grid(True, alpha=0.3)
    
    # ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # ax2.xaxis.set_major_locator(mdates.MonthLocator())
    # plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # 統計情報
    print(f"\\n=== キロ {target_kilo:.3f}±{kilo_range:.3f}m の最大振幅統計 ===")
    print(f"データセット数: {len(max_amp_df)}")
    print(f"振幅範囲: {max_amp_df['max_amplitude'].min():.4f} - {max_amp_df['max_amplitude'].max():.4f}")
    print(f"平均振幅: {max_amp_df['max_amplitude'].mean():.4f}")
    print(f"標準偏差: {max_amp_df['max_amplitude'].std():.4f}")
    
    # 波形プロット
    print("\\n波形連続プロットを生成中...")
    plot_waveforms_with_amplitude_range(data0_up_res, corrected_datasets, amplitude_df, dataset_info, reference_path,
                                      target_kilo, kilo_range, 
                                      plot_kilo_start, plot_kilo_end, figsize=figsize, 
                                      show_original=show_original)
    
    return max_amp_df