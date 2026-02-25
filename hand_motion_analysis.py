# hand_motion_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import os
from pathlib import Path
import warnings
import zipfile
# ----------------------------
# 1. 样本熵（Sample Entropy）
# ----------------------------
@jit(nopython=True)

def sample_entropy(time_series, m=2, r=0.2):
    N = len(time_series)
    if N <= m + 1:
        return np.nan
    std_ts = np.std(time_series)
    if std_ts == 0:
        return np.nan
    r = r * std_ts
    count1 = 0
    count2 = 0
    for i in range(N - m):
        for j in range(i + 1, N - m):
            if np.max(np.abs(time_series[i:i+m] - time_series[j:j+m])) < r:
                count1 += 1
                if np.abs(time_series[i+m] - time_series[j+m]) < r:
                    count2 += 1
    if count1 == 0 or count2 == 0:
        return np.nan
    return -np.log(count2 / count1)

def coarse_grain(time_series, scale):
    """对时间序列进行粗粒化"""
    N = len(time_series)
    if scale == 1:
        return time_series
    if N % scale != 0:
        trimmed_len = (N // scale) * scale
        time_series = time_series[:trimmed_len]
    return np.mean(time_series.reshape(-1, scale), axis=1)          #把数组重塑一个二维数组，每行有scale个元素===-1表示“自动推断行数”（注意：行数必须与scale整除）

def multiscale_entropy(time_series, max_scale=10, m=2, r=0.2):
    """计算多尺度熵（返回长度为 max_scale 的数组）"""
    mse = np.full(max_scale, np.nan)
    for s in range(1, max_scale + 1):
        cg = coarse_grain(time_series, s)
        if len(cg) > m + 1:
            #r = r_ratio * np.std(cg)           #为什么不这样做
            mse[s - 1] = sample_entropy(cg, m=m, r=r)
    return mse

# ----------------------------
# 3. 数据加载
# ----------------------------
def load_data(file_path, n_joints):
    df = pd.read_excel(file_path, engine='openpyxl')
    required_cols = ['frameID', 'Time']
    for i in range(1, n_joints + 1):
        required_cols.extend([f'centroidX_{i}', f'centroidY_{i}'])
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    coords = np.stack([
        df[[f'centroidX_{i}', f'centroidY_{i}']].values 
        for i in range(1, n_joints + 1)             #这里的for循环影响上面---把DataFrame转为numpy数组
    ], axis=1)  # (N, 8, 2)
    times = df['Time'].values

    if n_joints == 9:
        warnings.warn(
            "检测到 9 个 ID 点：已将 ID1 作为标准参考点并从分析中移除，"
            "所有关节编号已整体前移一个（原 ID2 → 新 ID1，…，原 ID9 → 新 ID8）。",
            UserWarning
        )
        # 以 ID1 为参考点，消除整体平移
        coords = coords - coords[:, 0:1, :]

        # 只保留 ID2–ID9（共 8 个关节）
        coords = coords[:, 1:, :]

        # 同步更新关节数
        n_joints = 8     # 使用第 0 个关节作为参考点（消除整体平移）


    return coords, times, df, n_joints

# ----------------------------
# 4. 特征计算（同前）
# ----------------------------
def compute_features(coords, times, lag=1):
    """
    基于统一 lag 窗口计算所有运动特征
    coords: (N, 8, 2)
    times: (N,)
    lag: int, 观测窗口（帧数）
    """
    N, n_joints, _ = coords.shape
    if lag >= N:
        raise ValueError("lag must be < number of frames")

    dt_global = times[1] - times[0]  # 假设均匀采样，≈1/60
    delta_t = lag * dt_global         # 实际时间间隔

    # --- 1. 位移到原点（不变）
    origin = coords[0]          #所有ID点的第一个X和Y的数据
    disp_origin = np.linalg.norm(coords - origin, axis=2)  # (N, 8)====计算每个点，每帧到第一个点的距离

    # --- 2. MSD (lag-based displacement)
    msd = np.full((N, n_joints), np.nan) #创建新数组
    msd[lag:] = np.linalg.norm(coords[lag:] - coords[:-lag], axis=2) #基于观察窗口       #在lag步长下的欧氏距离

    # --- 3. 速度（基于 lag 窗口）
    velocities_lag = np.full((N, n_joints, 2), np.nan)
    velocities_lag[lag:] = (coords[lag:] - coords[:-lag]) / delta_t  # (m/s)    #速度计算    #？？单位是像素还是m
    speeds = np.full((N, n_joints), np.nan)
    speeds[lag:] = np.linalg.norm(velocities_lag[lag:], axis=2)         #在lag步长下的，速率计算

    # --- 4. 运动方向角（基于 lag 位移向量）         #逐帧计算的运动方向角（瞬时方向）
    motion_angles = np.full((N, n_joints), np.nan)
    dx = coords[lag:, :, 0] - coords[:-lag, :, 0]  # (N-lag, 8)         #每个关节在 lag 时间步内的 x 方向位移 dx
    dy = coords[lag:, :, 1] - coords[:-lag, :, 1]
    motion_angles[lag:] = np.arctan2(dy, dx)        #单位：弧度      #以左上角为（0，0），x向右为正方向，y向下为正方向！！！！！

    # --- 5. 角速度（angular velocity）：需对 angle 序列再做 lag 差分
    angular_vel = np.full((N, n_joints), np.nan)
    if N > 2 * lag:
        angles_valid = motion_angles[lag:]  # (N-lag, 8)
        angle_diff = angles_valid[lag:] - angles_valid[:-lag]  # (N-2*lag, 8)       #在lag 时间步内的角度变化量
        # 处理角度环绕 [-π, π]
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi     #得到最小旋转角度
        angular_vel[2 * lag:] = angle_diff / delta_t  # rad/s

    # --- 6. 加速度（acceleration）和 Jerk（可选）
    acceleration = np.full((N, n_joints, 2), np.nan)
    jerk = np.full((N, n_joints), np.nan)

    if N > 2 * lag:
        # acceleration ≈ (v[t+lag] - v[t]) / delta_t
        v_valid = velocities_lag[lag:]  # (N-lag, 8, 2)
        acc_valid = (v_valid[lag:] - v_valid[:-lag]) / delta_t  # (N-2*lag, 8, 2)       #平均加速度=速度变化量/lag 步对应的实际时间
        acceleration[2 * lag:] = acc_valid          #加速度向量
        jerk[2 * lag:] = np.linalg.norm(acc_valid, axis=2)  # 这里用 |acc| 代替 jerk（更稳定）        #加速度大小

        # 若需真实 jerk（加速度的变化率），需 3*lag，此处简化
        # jerk = np.full(...); jerk[3*lag:] = ...

    return {
        'disp_origin': disp_origin,
        'msd': msd,
        'speeds': speeds,
        'motion_angles': motion_angles,
        'angular_vel': angular_vel,
        'jerk': jerk,          # 实际为 |acceleration|
        'times': times,
        'lag': lag,
        'delta_t': delta_t
    }

def compute_entropy_matrices(coords, n_joints, max_lag=15):
    velocities = np.diff(coords, axis=0)
    speed_seq = np.linalg.norm(velocities, axis=2)

    se_matrix = np.full((n_joints, max_lag), np.nan)
    mse_matrix = np.full((n_joints, max_lag), np.nan)

    for j in range(n_joints):
        ts = speed_seq[:, j]
        if len(ts) < 30:
            continue

        for lag in range(1, max_lag + 1):
            if len(ts) - lag < 20:
                break
            msd_ts = np.linalg.norm(
                coords[lag:, j] - coords[:-lag, j], axis=1
            )
            se_matrix[j, lag - 1] = sample_entropy(msd_ts, m=2, r=0.2)

        mse_vals = multiscale_entropy(ts, max_scale=max_lag, m=2, r=0.15)
        mse_matrix[j, :len(mse_vals)] = mse_vals

    return se_matrix, mse_matrix, speed_seq

def feature_plot(features, lag_selected, n_joints, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t = features['times']

    metrics = {
        'Displacement_from_Origin': features['disp_origin'],
        f'MSD_lag{lag_selected}': features['msd'],
        f'Speed_lag{lag_selected}': features['speeds'],
        f'Motion_Angles_lag{lag_selected}': features['motion_angles'],
        f'Angular_Velocity_lag{lag_selected}': features['angular_vel'],
        f'Acceleration_lag{lag_selected}': features['jerk']
    }

    for metric_name, data in metrics.items():
        valid_data = data[~np.isnan(data)]
        if len(valid_data) == 0:
            y_min, y_max = 0, 1
        else:
            y_min, y_max = np.nanmin(data), np.nanmax(data)
            margin = (y_max - y_min) * 0.05
            y_min -= margin
            y_max += margin

        fig, axes = plt.subplots(n_joints, 1, figsize=(10, 2*n_joints), sharex=True)
        fig.suptitle(metric_name, fontsize=14)

        for j in range(n_joints):
            axes[j].plot(t, data[:, j])
            axes[j].set_ylabel(f'J{j+1}')
            axes[j].set_ylim(y_min, y_max)
            axes[j].grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        plt.savefig(out_dir / f"{metric_name}.png", dpi=150)
        plt.savefig(out_dir / f"{metric_name}.svg", dpi=150)
        plt.close(fig)   # ❗关键

def corr_plot(features, lag_selected, n_joints, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def draw_corr(data, title, fname):
        corr = np.corrcoef(data.T)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax)

        ax.set_title(title)
        ticks = [f'J{i+1}' for i in range(n_joints)]
        ax.set_xticks(range(n_joints))
        ax.set_yticks(range(n_joints))
        ax.set_xticklabels(ticks)
        ax.set_yticklabels(ticks)

        for i in range(n_joints):
            for j in range(n_joints):
                ax.text(j, i, f"{corr[i,j]:.2f}",
                        ha="center", va="center",
                        color="white" if abs(corr[i,j]) > 0.5 else "black")

        plt.tight_layout()
        plt.savefig(out_dir / f"{fname}.png", dpi=150)
        plt.savefig(out_dir / f"{fname}.svg", dpi=150)
        plt.close(fig)

    draw_corr(features['speeds'][lag_selected:], 
              f"Speed Correlation (lag={lag_selected})",
              "corr_speed")

    draw_corr(features['motion_angles'][lag_selected:], 
              f"Motion Angle Correlation (lag={lag_selected})",
              "corr_motion_angle")

    draw_corr(features['angular_vel'][2*lag_selected:], 
              f"Angular Velocity Correlation (lag={lag_selected})",
              "corr_angular_velocity")

    draw_corr(features['jerk'][2*lag_selected:], 
              f"Jerk Correlation (lag={lag_selected})",
              "corr_jerk")


def zip_results_images_only(fig_dir, zip_path):
    fig_dir = Path(fig_dir)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in fig_dir.rglob("*"):
            if file.is_file() and file.suffix.lower() in [".png", ".svg"]:
                # relative_to 保证 zip 内部路径不包含 fig_dir 上层
                zipf.write(file, file.relative_to(fig_dir))





def main():
    # ===== 你只需要改这几个参数 =====
    file_path = Path(__file__).parent / "data.xlsx"
    n_joints = 8
    lag = 5
    max_lag = 15
    coords, times, df, n_joints = load_data(file_path, n_joints)

    # 熵
    se_matrix, mse_matrix, speed_seq = compute_entropy_matrices(
        coords, n_joints, max_lag=max_lag
    )

    # 运动学特征
    features = compute_features(coords, times, lag=lag)

    # ===== 写 Excel =====
    out_path = Path(file_path).parent / "results.xlsx"

    with pd.ExcelWriter(out_path) as writer:
        # coords
        df_coords = pd.DataFrame(
            coords.reshape(coords.shape[0], -1),
            columns=[
                f"ID{i+1}_{axis}"
                for i in range(coords.shape[1])
                for axis in ["X", "Y"]
            ]
        )
        df_coords.insert(0, "Time", times)
        df_coords.to_excel(writer, sheet_name="coords_xy", index=False)

        pd.DataFrame(speed_seq).to_excel(writer, "speed", index=False)
        pd.DataFrame(se_matrix).to_excel(writer, "sample_entropy", index=False)
        pd.DataFrame(mse_matrix).to_excel(writer, "mse", index=False)

        pd.DataFrame(features["disp_origin"]).to_excel(writer, "disp_origin", index=False)
        pd.DataFrame(features["msd"]).to_excel(writer, "msd", index=False)
        pd.DataFrame(features["speeds"]).to_excel(writer, "speeds", index=False)
        pd.DataFrame(features["motion_angles"]).to_excel(writer, "motion_angles", index=False)
        pd.DataFrame(features["angular_vel"]).to_excel(writer, "angular_velocity", index=False)
        pd.DataFrame(features["jerk"]).to_excel(writer, "jerk", index=False)

    print("✅ 分析完成，仅生成 Excel，可完全复现")



def run_analysis(
    file_path,
    n_joints=8,
    lag=2,
    max_lag=15,
    out_dir=None
):
    """
    Flask / CLI 通用的分析入口
    不交互、不画图、只产出 Excel
    """

    file_path = Path(file_path)

    if out_dir is None:
        out_dir = file_path.parent
    else:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 读数据
    coords, times, df_raw, n_joints = load_data(file_path, n_joints)

    # 2. 熵分析
    se_matrix, mse_matrix, speed_seq = compute_entropy_matrices(
        coords, n_joints, max_lag=max_lag
    )

    # 3. 运动学特征
    features = compute_features(coords, times, lag=lag)

    # 4. 写 Excel
    out_path = out_dir / "results.xlsx"

    fig_dir = out_dir / "figures"
    feature_plot(
        features,
        lag_selected=lag,
        n_joints=n_joints,
        out_dir=fig_dir / "features"
    )

    corr_plot(
        features,
        lag_selected=lag,
        n_joints=n_joints,
        out_dir=fig_dir / "correlations"
    )
    zip_path = out_dir / "results.zip"
    zip_results_images_only(fig_dir, zip_path)


    with pd.ExcelWriter(out_path) as writer:

        # coords
        df_coords = pd.DataFrame(
            coords.reshape(coords.shape[0], -1),
            columns=[
                f"ID{i+1}_{axis}"
                for i in range(coords.shape[1])
                for axis in ["X", "Y"]
            ]
        )
        df_coords.insert(0, "Time", times)
        df_coords.to_excel(writer, sheet_name="coords_xy", index=False)

        pd.DataFrame(speed_seq).to_excel(writer, "speed", index=False)
        pd.DataFrame(se_matrix).to_excel(writer, "sample_entropy", index=False)
        pd.DataFrame(mse_matrix).to_excel(writer, "mse", index=False)

        pd.DataFrame(features["disp_origin"]).to_excel(writer, "disp_origin", index=False)
        pd.DataFrame(features["msd"]).to_excel(writer, "msd", index=False)
        pd.DataFrame(features["speeds"]).to_excel(writer, "speeds", index=False)
        pd.DataFrame(features["motion_angles"]).to_excel(writer, "motion_angles", index=False)
        pd.DataFrame(features["angular_vel"]).to_excel(writer, "angular_velocity", index=False)
        pd.DataFrame(features["jerk"]).to_excel(writer, "jerk", index=False)

    return {
        "excel_path": str(out_path),
        "n_frames": coords.shape[0],
        "n_joints": n_joints,
        "lag": lag,
        "excel_path": str(out_path),
        "zip_path": str(zip_path)
    }
        


if __name__ == "__main__":
    main()