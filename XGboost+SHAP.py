import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
import os
from scipy import stats
from matplotlib.patches import Rectangle

# ====================================
#         设置字体 Times New Roman
# ====================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15

# ====================================
#             读取数据
# ====================================
df = pd.read_csv("熔池深度和粗糙度.csv")

X = df.iloc[:, 0:3]  # 功率、速度、激光间隔
y_depth = df.iloc[:, 3]
y_rough = df.iloc[:, 4]
y_pore = df.iloc[:, 5]

# 分类编码
le = LabelEncoder()
y_pore_encoded = le.fit_transform(y_pore)

# 获取类别名称
class_names = le.classes_
n_classes = len(class_names)
print(f"检测到 {n_classes} 个孔隙类别: {class_names}")

# ====================================
#           XGB 参数
# ====================================
params = {
    "booster": "gbtree",
    "n_estimators": 100,
    "learning_rate": 0.1,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "subsample": 1,
    "colsample_bytree": 1,
    "colsample_bynode": 1,
    "min_child_weight": 0,
    "max_depth": 8,
    "random_state": 0
}

# 输出目录
output_dir = "shap_output29"
os.makedirs(output_dir, exist_ok=True)


# =====================================================================
#                 分析ICE异常点的函数（改进版）
# =====================================================================
def analyze_ice_anomalies(model, X, feature_name, model_name, is_classification=False, target_class=None):
    """
    分析ICE图中的异常点及其对应的参数区域

    参数:
    model: 训练好的模型
    X: 特征数据
    feature_name: 要分析的特征名
    model_name: 模型名称
    is_classification: 是否为分类模型
    target_class: 分类模型的目标类别
    """

    print(f"  分析{feature_name}的ICE异常点...")

    # 创建异常分析子目录
    anomaly_dir = os.path.join(output_dir, "ice_anomaly_analysis")
    os.makedirs(anomaly_dir, exist_ok=True)

    # 获取要分析的特征索引
    feature_idx = list(X.columns).index(feature_name)

    # 使用更可靠的方法计算ICE和PDP
    print(f"    使用sklearn的PartialDependenceDisplay计算ICE和PDP...")

    try:
        # 使用sklearn的PartialDependenceDisplay计算ICE和PDP
        fig, ax = plt.subplots(figsize=(10, 7))

        if is_classification and target_class is not None:
            # 分类模型
            disp = PartialDependenceDisplay.from_estimator(
                model, X, features=[feature_name],
                kind='both',
                target=target_class,
                ax=ax,
                pd_line_kw={"color": "red", "linewidth": 2},
                ice_lines_kw={"color": "blue", "alpha": 0.1, "linewidth": 0.5}
            )
        else:
            # 回归模型
            disp = PartialDependenceDisplay.from_estimator(
                model, X, features=[feature_name],
                kind='both',
                ax=ax,
                pd_line_kw={"color": "red", "linewidth": 2},
                ice_lines_kw={"color": "blue", "alpha": 0.1, "linewidth": 0.5}
            )

        # 从PartialDependenceDisplay对象获取数据
        # 注意：sklearn的PartialDependenceDisplay内部数据结构可能因版本而异
        # 我们需要手动提取ICE和PDP数据

        # 方法1：尝试从disp对象获取数据
        try:
            # 对于sklearn >= 1.0版本
            pdp_values = disp.pd_results[0]['average']
            ice_values = disp.pd_results[0]['individual']
            grid_points = disp.pd_results[0]['values'][0]
        except:
            # 如果上面的方法失败，使用手动计算方法
            print(f"    无法从PartialDependenceDisplay提取数据，使用手动计算...")

            # 生成网格点
            grid_resolution = min(50, len(np.unique(X.iloc[:, feature_idx])))
            grid_points = np.unique(X.iloc[:, feature_idx])
            if len(grid_points) > grid_resolution:
                # 如果唯一值太多，进行采样
                grid_points = np.linspace(X.iloc[:, feature_idx].min(),
                                          X.iloc[:, feature_idx].max(),
                                          grid_resolution)

            # 手动计算ICE值
            n_samples = min(200, len(X))
            if len(X) > n_samples:
                sample_indices = np.random.choice(len(X), n_samples, replace=False)
                X_sample = X.iloc[sample_indices].copy()
            else:
                sample_indices = np.arange(len(X))
                X_sample = X.copy()

            ice_values = []
            for i in range(len(X_sample)):
                if i % 50 == 0 and i > 0:
                    print(f"      正在计算ICE值 {i}/{len(X_sample)}...")

                ice_curve = []
                original_value = X_sample.iloc[i, feature_idx]

                for grid_point in grid_points:
                    X_temp = X_sample.copy()
                    X_temp.iloc[i, feature_idx] = grid_point

                    if is_classification:
                        if target_class is not None:
                            pred = model.predict_proba(X_temp.iloc[i:i + 1])[0, target_class]
                        else:
                            pred = model.predict_proba(X_temp.iloc[i:i + 1])[0, 0]  # 默认第一个类别
                    else:
                        pred = model.predict(X_temp.iloc[i:i + 1])[0]

                    ice_curve.append(pred)

                ice_values.append(ice_curve)

            ice_values = np.array(ice_values)

            # 计算PDP值（ICE值的平均值）
            pdp_values = ice_values.mean(axis=0)

        # 确保数据形状正确
        if ice_values.shape[1] != len(grid_points):
            print(f"    数据形状不匹配: ice_values.shape={ice_values.shape}, grid_points.shape={grid_points.shape}")
            # 尝试转置ice_values
            if ice_values.shape[0] == len(grid_points):
                ice_values = ice_values.T
                print(f"    已转置ice_values: 新形状={ice_values.shape}")

        # 现在计算异常点
        if ice_values.shape[1] == len(grid_points) and len(pdp_values) == len(grid_points):
            # 计算每条ICE曲线与PDP曲线的差异
            ice_differences = np.abs(ice_values - pdp_values)
            max_differences = ice_differences.max(axis=1)  # 每个样本的最大差异
            mean_differences = ice_differences.mean(axis=1)  # 每个样本的平均差异

            # 使用Z-score识别异常ICE曲线
            z_scores = stats.zscore(max_differences)
            anomaly_threshold = 2.0  # Z-score阈值
            anomaly_indices = np.where(np.abs(z_scores) > anomaly_threshold)[0]

            # 使用IQR方法识别异常
            Q1 = np.percentile(max_differences, 25)
            Q3 = np.percentile(max_differences, 75)
            IQR = Q3 - Q1
            iqr_anomaly_indices = np.where(
                (max_differences < (Q1 - 1.5 * IQR)) |
                (max_differences > (Q3 + 1.5 * IQR))
            )[0]

            # 合并两种方法检测到的异常
            all_anomaly_indices = np.unique(np.concatenate([anomaly_indices, iqr_anomaly_indices]))

            print(f"    检测到 {len(all_anomaly_indices)} 条异常ICE曲线")

            if len(all_anomaly_indices) > 0:
                # 创建异常分析图
                create_anomaly_analysis_plot(
                    ice_values, pdp_values, grid_points, X_sample,
                    all_anomaly_indices, max_differences, mean_differences,
                    z_scores, anomaly_indices, iqr_anomaly_indices,
                    feature_name, model_name, anomaly_dir
                )

                # 保存异常点的详细数据
                save_anomaly_details(
                    X_sample, all_anomaly_indices, max_differences,
                    mean_differences, z_scores, anomaly_indices,
                    iqr_anomaly_indices, feature_name, model_name,
                    anomaly_dir, sample_indices
                )
            else:
                print(f"    未检测到显著的ICE异常点")

        plt.close(fig)  # 关闭之前创建的图形

    except Exception as e:
        print(f"    ICE异常点分析出错: {e}")
        import traceback
        traceback.print_exc()

    return []


def create_anomaly_analysis_plot(ice_values, pdp_values, grid_points, X_sample,
                                 all_anomaly_indices, max_differences, mean_differences,
                                 z_scores, anomaly_indices, iqr_anomaly_indices,
                                 feature_name, model_name, anomaly_dir):
    """创建异常点分析图"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{model_name} - ICE Anomaly Analysis for {feature_name}', fontsize=16)

    # 1. 显示所有ICE曲线，高亮异常曲线
    ax1 = axes[0, 0]
    for i in range(len(ice_values)):
        if i in all_anomaly_indices:
            ax1.plot(grid_points, ice_values[i], color='red', alpha=0.5, linewidth=1.0)
        else:
            ax1.plot(grid_points, ice_values[i], color='blue', alpha=0.1, linewidth=0.5)

    # 绘制PDP曲线
    ax1.plot(grid_points, pdp_values, color='black', linewidth=3, label='PDP')
    ax1.set_xlabel(feature_name)
    ax1.set_ylabel('Predicted Value')
    ax1.set_title(f'ICE Curves (Red=Anomalies, N={len(all_anomaly_indices)})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 异常样本的特征分布
    ax2 = axes[0, 1]
    feature_idx = list(X_sample.columns).index(feature_name)
    all_values = X_sample.iloc[:, feature_idx].values
    anomaly_values = X_sample.iloc[all_anomaly_indices, feature_idx].values if len(all_anomaly_indices) > 0 else []

    # 绘制所有样本的分布
    ax2.hist(all_values, bins=30, alpha=0.5, color='blue', label='All Samples', density=True)
    # 绘制异常样本的分布
    if len(anomaly_values) > 0:
        ax2.hist(anomaly_values, bins=15, alpha=0.7, color='red', label='Anomalies', density=True)

    ax2.set_xlabel(feature_name)
    ax2.set_ylabel('Density')
    ax2.set_title(f'Feature Distribution of Anomalies')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 异常点在特征空间中的分布
    ax3 = axes[0, 2]
    other_features = [col for col in X_sample.columns if col != feature_name]
    if len(other_features) > 0 and len(all_anomaly_indices) > 0:
        other_feature = other_features[0]
        other_idx = list(X_sample.columns).index(other_feature)

        # 绘制所有样本
        ax3.scatter(X_sample.iloc[:, feature_idx], X_sample.iloc[:, other_idx],
                    alpha=0.3, color='blue', s=10, label='All Samples')
        # 绘制异常样本
        ax3.scatter(X_sample.iloc[all_anomaly_indices, feature_idx],
                    X_sample.iloc[all_anomaly_indices, other_idx],
                    alpha=0.8, color='red', s=50, label='Anomalies', edgecolors='black')

        ax3.set_xlabel(feature_name)
        ax3.set_ylabel(other_feature)
        ax3.set_title(f'Anomalies in Feature Space')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No anomalies or only one feature',
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Feature Space')
        ax3.axis('off')

    # 4. 异常点的统计特征分布
    ax4 = axes[1, 0]
    if len(all_anomaly_indices) > 0:
        normal_indices = np.setdiff1d(np.arange(len(X_sample)), all_anomaly_indices)

        if len(normal_indices) > 0:
            box_data = [max_differences[normal_indices], max_differences[all_anomaly_indices]]
            box_labels = ['Normal', 'Anomaly']

            bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('lightcoral')

            ax4.set_ylabel('Max ICE Difference from PDP')
            ax4.set_title('Statistical Difference: Normal vs Anomaly')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Insufficient normal samples',
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Statistical Analysis')
            ax4.axis('off')
    else:
        ax4.text(0.5, 0.5, 'No anomalies detected',
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Statistical Analysis')
        ax4.axis('off')

    # 5. 异常点在不同参数区间的分布
    ax5 = axes[1, 1]
    if len(all_anomaly_indices) > 0:
        n_bins = min(5, len(np.unique(all_values)))
        if n_bins > 1:
            bins = np.linspace(min(all_values), max(all_values), n_bins + 1)

            bin_counts = []
            anomaly_counts = []

            for i in range(n_bins):
                mask = (all_values >= bins[i]) & (all_values < bins[i + 1])
                if i == n_bins - 1:
                    mask = (all_values >= bins[i]) & (all_values <= bins[i + 1])

                total_in_bin = mask.sum()
                anomalies_in_bin = 0
                if len(all_anomaly_indices) > 0:
                    anomaly_mask = mask[all_anomaly_indices] if len(all_anomaly_indices) < len(mask) else mask
                    anomalies_in_bin = anomaly_mask.sum()

                bin_counts.append(total_in_bin)
                anomaly_counts.append(anomalies_in_bin)

            x_pos = np.arange(n_bins)
            width = 0.35

            ax5.bar(x_pos - width / 2, bin_counts, width, label='Total Samples', color='blue', alpha=0.6)
            ax5.bar(x_pos + width / 2, anomaly_counts, width, label='Anomalies', color='red', alpha=0.6)

            ax5.set_xlabel(f'{feature_name} Bins')
            ax5.set_ylabel('Count')
            ax5.set_title('Anomaly Distribution Across Feature Bins')
            ax5.set_xticks(x_pos)
            ax5.set_xticklabels([f'Bin {i + 1}' for i in range(n_bins)])
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'Insufficient unique values for binning',
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax5.transAxes, fontsize=12)
            ax5.set_title('Binned Analysis')
            ax5.axis('off')
    else:
        ax5.text(0.5, 0.5, 'No anomalies detected',
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax5.transAxes, fontsize=12)
        ax5.set_title('Binned Analysis')
        ax5.axis('off')

    # 6. 异常点信息展示
    ax6 = axes[1, 2]
    if len(all_anomaly_indices) > 0:
        # 显示异常点统计信息
        info_text = f"Total anomalies: {len(all_anomaly_indices)}\n"
        info_text += f"Z-score anomalies: {len(anomaly_indices)}\n"
        info_text += f"IQR anomalies: {len(iqr_anomaly_indices)}\n\n"

        if len(all_anomaly_indices) <= 10:
            info_text += "Anomaly indices:\n"
            info_text += ", ".join([str(i) for i in all_anomaly_indices[:10]])
        else:
            info_text += f"Top 10 anomaly indices:\n"
            info_text += ", ".join([str(i) for i in all_anomaly_indices[:10]])

        ax6.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center')
        ax6.set_title('Anomaly Information')
        ax6.axis('off')
    else:
        ax6.text(0.5, 0.5, 'No anomalies detected',
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax6.transAxes, fontsize=12)
        ax6.set_title('Anomaly Information')
        ax6.axis('off')

    plt.tight_layout()
    plt.savefig(f"{anomaly_dir}/{model_name}_anomaly_analysis_{feature_name}.png",
                dpi=300, bbox_inches="tight")
    plt.close()


def save_anomaly_details(X_sample, all_anomaly_indices, max_differences,
                         mean_differences, z_scores, anomaly_indices,
                         iqr_anomaly_indices, feature_name, model_name,
                         anomaly_dir, sample_indices):
    """保存异常点详细信息"""

    if len(all_anomaly_indices) > 0:
        anomaly_df = X_sample.iloc[all_anomaly_indices].copy()
        anomaly_df['max_ice_difference'] = max_differences[all_anomaly_indices]
        anomaly_df['mean_ice_difference'] = mean_differences[all_anomaly_indices]
        anomaly_df['z_score'] = z_scores[all_anomaly_indices]
        anomaly_df['is_iqr_anomaly'] = np.isin(all_anomaly_indices, iqr_anomaly_indices)

        # 添加原始样本索引
        anomaly_df['original_sample_index'] = sample_indices[all_anomaly_indices]

        anomaly_df.to_csv(f"{anomaly_dir}/{model_name}_anomaly_details_{feature_name}.csv", index=False)
        print(f"    异常点详细信息已保存到: {anomaly_dir}/{model_name}_anomaly_details_{feature_name}.csv")

        # 打印统计信息
        print(f"    {feature_name}异常点的参数统计:")
        print(anomaly_df.describe())


# =====================================================================
#                 绘制PDP和ICE图的函数
# =====================================================================
def plot_pdp_ice(model, X, y, model_name, is_classification=False):
    """
    绘制PDP（部分依赖图）和ICE（个体条件期望图）

    参数:
    model: 训练好的模型
    X: 特征数据
    y: 目标变量
    model_name: 模型名称（用于保存文件）
    is_classification: 是否为分类模型
    """

    print(f"\n===== 为 {model_name} 生成PDP和ICE图 =====")

    # 为PDP/ICE创建子目录
    pdp_ice_dir = os.path.join(output_dir, "pdp_ice")
    os.makedirs(pdp_ice_dir, exist_ok=True)

    # 获取特征名称
    feature_names = X.columns.tolist()
    n_features = len(feature_names)

    # ===========================================================
    # 对于回归模型
    # ===========================================================
    if not is_classification:
        print(f"  生成PDP图...")

        # 创建图形
        fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 5))

        if n_features == 1:
            axes = [axes]

        for i, feature in enumerate(feature_names):
            ax = axes[i]

            # 绘制PDP图
            PartialDependenceDisplay.from_estimator(
                model, X, features=[feature],
                ax=ax, line_kw={"color": "red", "linewidth": 2.5}
            )

            ax.set_xlabel(feature, fontsize=12)
            ax.set_ylabel('Partial Dependence', fontsize=12)
            ax.set_title(f'PDP for {feature}', fontsize=14)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{pdp_ice_dir}/{model_name}_pdp.png", dpi=300, bbox_inches="tight")
        plt.close()

        # ===========================================================
        # 绘制ICE图并分析异常点
        # ===========================================================
        print(f"  生成ICE图并分析异常点...")

        # 对每个特征绘制ICE图并分析异常点
        for i, feature in enumerate(feature_names):
            try:
                # 创建图形
                fig, ax = plt.subplots(figsize=(10, 7))

                # 计算PDP和ICE值
                disp = PartialDependenceDisplay.from_estimator(
                    model, X, features=[feature],
                    kind='both',
                    ax=ax,
                    pd_line_kw={"color": "red", "linewidth": 3, "label": "PDP"},
                    ice_lines_kw={"color": "blue", "alpha": 0.1, "linewidth": 0.5}
                )

                # 添加图例
                ax.legend(fontsize=10)

                # 添加标签和标题
                ax.set_xlabel(feature, fontsize=12)
                ax.set_ylabel('Predicted Value', fontsize=12)
                ax.set_title(f'{model_name} - ICE Plot for {feature}', fontsize=14)
                ax.grid(True, alpha=0.3)

                # 保存图形
                plt.tight_layout()
                plt.savefig(f"{pdp_ice_dir}/{model_name}_ice_{feature}.png",
                            dpi=300, bbox_inches="tight")
                plt.close()

                # 分析该特征的ICE异常点
                analyze_ice_anomalies(model, X, feature, model_name, is_classification=False)

            except Exception as e:
                print(f"    绘制{feature}的ICE图出错: {e}")
                import traceback
                traceback.print_exc()
                continue

        # ===========================================================
        # 绘制所有特征的ICE图（子图形式）
        # ===========================================================
        print(f"  生成组合ICE图...")

        try:
            fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 5))

            if n_features == 1:
                axes = [axes]

            for i, feature in enumerate(feature_names):
                ax = axes[i]

                # 回归模型
                PartialDependenceDisplay.from_estimator(
                    model, X, features=[feature],
                    kind='both',
                    ax=ax,
                    pd_line_kw={"color": "red", "linewidth": 2.5},
                    ice_lines_kw={"color": "blue", "alpha": 0.15, "linewidth": 0.6}
                )

                ax.set_xlabel(feature, fontsize=11)
                ax.set_ylabel('Predicted Value', fontsize=11)
                ax.set_title(f'ICE for {feature}', fontsize=12)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f"{pdp_ice_dir}/{model_name}_ice_combined.png",
                        dpi=300, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"    生成组合ICE图出错: {e}")

    # ===========================================================
    # 对于分类模型
    # ===========================================================
    else:
        # 获取类别数量
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        print(f"  分类模型有 {n_classes} 个类别")

        # 为每个类别生成PDP图
        print(f"  为每个类别生成PDP图...")

        for class_idx in unique_classes:
            fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 5))

            if n_features == 1:
                axes = [axes]

            for i, feature in enumerate(feature_names):
                ax = axes[i]

                # 绘制PDP图，指定目标类别
                try:
                    PartialDependenceDisplay.from_estimator(
                        model, X, features=[feature],
                        target=class_idx,
                        ax=ax, line_kw={"color": "red", "linewidth": 2.5}
                    )
                except Exception as e:
                    print(f"    警告: 无法为类别 {class_idx} 绘制PDP图: {e}")
                    continue

                # 添加标签和标题
                ax.set_xlabel(feature, fontsize=12)
                ax.set_ylabel('Partial Dependence', fontsize=12)
                ax.set_title(f'PDP for {feature} (Class {class_idx})', fontsize=14)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f"{pdp_ice_dir}/{model_name}_pdp_class_{class_idx}.png",
                        dpi=300, bbox_inches="tight")
            plt.close()

        # 为每个类别和每个特征绘制ICE图
        print(f"  为每个类别生成ICE图...")

        for class_idx in unique_classes:
            for feature in feature_names:
                # 创建图形
                fig, ax = plt.subplots(figsize=(10, 7))

                try:
                    # 计算PDP和ICE值
                    PartialDependenceDisplay.from_estimator(
                        model, X, features=[feature],
                        kind='both',
                        target=class_idx,
                        ax=ax,
                        pd_line_kw={"color": "red", "linewidth": 3, "label": "PDP"},
                        ice_lines_kw={"color": "blue", "alpha": 0.1, "linewidth": 0.5}
                    )

                    # 添加图例
                    ax.legend(fontsize=10)

                    # 添加标签和标题
                    ax.set_xlabel(feature, fontsize=12)
                    ax.set_ylabel('Predicted Probability', fontsize=12)
                    ax.set_title(f'{model_name} - ICE Plot for {feature} (Class {class_idx})',
                                 fontsize=14)
                    ax.grid(True, alpha=0.3)

                    # 保存图形
                    plt.tight_layout()
                    plt.savefig(f"{pdp_ice_dir}/{model_name}_ice_{feature}_class_{class_idx}.png",
                                dpi=300, bbox_inches="tight")
                    plt.close()

                except Exception as e:
                    print(f"    警告: 无法为类别 {class_idx} 和特征 {feature} 绘制ICE图: {e}")

        # 为每个类别绘制组合ICE图
        print(f"  生成组合ICE图...")

        for class_idx in unique_classes:
            fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 5))

            if n_features == 1:
                axes = [axes]

            for i, feature in enumerate(feature_names):
                ax = axes[i]

                try:
                    # 分类模型：绘制指定类别的ICE图
                    PartialDependenceDisplay.from_estimator(
                        model, X, features=[feature],
                        kind='both',
                        target=class_idx,
                        ax=ax,
                        pd_line_kw={"color": "red", "linewidth": 2.5},
                        ice_lines_kw={"color": "blue", "alpha": 0.15, "linewidth": 0.6}
                    )

                    ax.set_xlabel(feature, fontsize=11)
                    ax.set_ylabel('Predicted Probability', fontsize=11)
                    ax.set_title(f'ICE for {feature} (Class {class_idx})', fontsize=12)
                    ax.grid(True, alpha=0.3)

                except Exception as e:
                    print(f"    警告: 无法为类别 {class_idx} 绘制组合ICE图: {e}")

            plt.tight_layout()
            plt.savefig(f"{pdp_ice_dir}/{model_name}_ice_combined_class_{class_idx}.png",
                        dpi=300, bbox_inches="tight")
            plt.close()

    print(f"  PDP和ICE图已保存到 {pdp_ice_dir}")


# =====================================================================
#                 通用模型 + SHAP 运行函数
# =====================================================================
def run_model_and_shap(model, X, y, name, is_classification=False):
    print(f"\n===== 运行模型：{name} =====")

    # 训练切分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, shuffle=True, random_state=0
    )

    # 5 折交叉验证
    cv_score = cross_val_score(model, X, y, cv=5)
    print(f"CV Mean Score = {cv_score.mean()}")

    # 模型训练
    model.fit(X_train, y_train)

    # SHAP
    try:
        explainer = shap.TreeExplainer(model)
        shap_raw = explainer.shap_values(X)

        # ===========================================================
        #           回归：shap_raw 是 numpy 数组
        #           分类：shap_raw 是 list，每类一个数组
        # ===========================================================
        if is_classification:
            # shap_raw 是 list
            num_classes = len(shap_raw)

            # 保存原始 list SHAP
            np.save(f"{output_dir}/{name}_shap_raw.npy", shap_raw, allow_pickle=True)

            # 每个类别保存单独 CSV
            for cls in range(num_classes):
                df_cls = pd.DataFrame(shap_raw[cls], columns=X.columns)
                df_cls.to_csv(f"{output_dir}/{name}_shap_values_class_{cls}.csv", index=False)

            # 用均值作为 overall shap
            shap_values = np.mean(np.array(shap_raw), axis=0)

        else:
            # 回归直接二维
            shap_values = shap_raw
            pd.DataFrame(shap_values, columns=X.columns).to_csv(
                f"{output_dir}/{name}_shap_values.csv", index=False
            )

        # ===========================================================
        #   1. 总体 SHAP summary 图
        # ===========================================================
        plt.figure()
        shap.summary_plot(shap_values, X, show=False)
        plt.title(f"{name} - SHAP Summary (Overall)")
        plt.savefig(f"{output_dir}/{name}_summary_overall.png", dpi=300, bbox_inches="tight")
        plt.close()

        # ===========================================================
        #   2. 总体 bar plot
        # ===========================================================
        plt.figure()
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.title(f"{name} - SHAP Bar Plot")
        plt.savefig(f"{output_dir}/{name}_bar_overall.png", dpi=300, bbox_inches="tight")
        plt.close()

        # ===========================================================
        #   3. 总体 heatmap
        # ===========================================================
        plt.figure(figsize=(6, 4))
        shap_mean = np.abs(shap_values).mean(axis=0)
        plt.imshow(shap_mean.reshape(1, -1), aspect="auto", cmap="viridis")
        plt.xticks(range(len(X.columns)), X.columns, rotation=45)
        plt.yticks([])
        plt.title(f"{name} - SHAP Heatmap")
        plt.colorbar()
        plt.savefig(f"{output_dir}/{name}_heatmap_overall.png", dpi=300, bbox_inches="tight")
        plt.close()

        # ===========================================================
        #   4. 总体 dependence plot - 使用单色
        # ===========================================================
        for feat in X.columns:
            plt.figure(figsize=(8, 6))

            # 使用 matplotlib 直接绘制单色散点图
            feature_idx = X.columns.get_loc(feat)
            plt.scatter(X.iloc[:, feature_idx], shap_values[:, feature_idx],
                        alpha=0.7, color='blue', s=20)
            plt.xlabel(feat, fontsize=12)
            plt.ylabel('SHAP value', fontsize=12)
            plt.title(f"{name} - SHAP Dependence ({feat})", fontsize=14)
            plt.grid(True, alpha=0.3)

            plt.savefig(f"{output_dir}/{name}_dependence_{feat}.png", dpi=300, bbox_inches="tight")
            plt.close()

        # ===========================================================
        #   5. 交互矩阵图（回归模型和分类模型都使用总体SHAP值）
        # ===========================================================
        try:
            # 尝试获取交互值
            if not is_classification:
                inter = explainer.shap_interaction_values(X)
                mean_inter = np.abs(inter).mean(axis=0)
            else:
                # 对于分类模型，使用第一个类别的交互值作为总体
                inter = explainer.shap_interaction_values(X)[0]
                mean_inter = np.abs(inter).mean(axis=0)

            plt.figure(figsize=(8, 6))
            im = plt.imshow(mean_inter, cmap="viridis")
            plt.colorbar(im)
            plt.xticks(range(len(X.columns)), X.columns, rotation=45)
            plt.yticks(range(len(X.columns)), X.columns)
            plt.title(f"{name} - SHAP Interaction Matrix")
            plt.savefig(f"{output_dir}/{name}_interaction_matrix.png",
                        dpi=300, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"无法生成交互矩阵图: {e}")

    except Exception as e:
        print(f"SHAP分析出错: {e}")
        import traceback
        traceback.print_exc()

    # ===========================================================
    #   6. 绘制PDP和ICE图，并分析异常点
    # ===========================================================
    plot_pdp_ice(model, X, y, name, is_classification)

    print(f"SHAP 和 PDP/ICE 分析已完成：{name}")


# =============================
#     运行三个模型
# =============================
print("=" * 50)
print("开始模型训练和可解释性分析")
print("=" * 50)

try:
    reg_depth = XGBRegressor(**params)
    run_model_and_shap(reg_depth, X, y_depth, "MeltPool_Depth")
except Exception as e:
    print(f"MeltPool_Depth模型分析出错: {e}")
    import traceback

    traceback.print_exc()

try:
    reg_rough = XGBRegressor(**params)
    run_model_and_shap(reg_rough, X, y_rough, "Surface_Roughness")
except Exception as e:
    print(f"Surface_Roughness模型分析出错: {e}")
    import traceback

    traceback.print_exc()

try:
    clf_pore = XGBClassifier(**params, use_label_encoder=False, eval_metric='mlogloss')
    run_model_and_shap(clf_pore, X, y_pore_encoded, "Pore_Type", is_classification=True)
except Exception as e:
    print(f"Pore_Type模型分析出错: {e}")
    import traceback

    traceback.print_exc()

print(f"\n" + "=" * 50)
print(f"🎉 全部模型与 SHAP、PDP、ICE 图生成完毕！")
print(f"📁 所有输出保存在: {output_dir}/")
print(f"📊 输出内容包括:")
print(f"   - SHAP 分析图（summary, bar, heatmap, dependence, interaction）")
print(f"   - PDP 图（部分依赖图）")
print(f"   - ICE 图（个体条件期望图）")
print(f"   - ICE异常点分析图和详细数据")
print(f"   - 每个特征的单独ICE图")
print(f"   - 所有特征的组合ICE图")
print(f"   - 分类模型为每个类别单独生成PDP/ICE图")
print(f"📈 ICE异常点分析包括:")
print(f"   1. 异常ICE曲线可视化")
print(f"   2. 异常点参数分布分析")
print(f"   3. 异常点在特征空间中的分布")
print(f"   4. 异常点统计特征分析")
print(f"   5. 异常点在不同参数区间的分布")
print(f"   6. 异常点的详细参数表格")
print("=" * 50)