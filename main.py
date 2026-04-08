# _*_coding:utf-8 _*_
import time
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

from P_GBG import get_gb_division_x
from QuantumMerge import merge_granules_with_quantum_proximity
from evaluation import evaluation

def format_time_str(seconds):
    time_struct = time.gmtime(seconds)
    time_str = time.strftime('%H:%M:%S', time_struct)
    micros = f"{seconds:.6f}".split('.')[-1]
    return f"{time_str}.{micros}"

def main():
    name_list = os.listdir('./datasets')
    name_list.sort() 
    
    for data_name in name_list:
        # 跳过非csv文件
        if not data_name.endswith('.csv'):
            continue
            
        np.random.seed(0)
        data_name_pure = data_name[:-4] # 去掉.csv后缀
        path = r'./datasets' + '/' + data_name

        df = pd.read_csv(path, header=None)
        X = df.values[:, 1:]
        y = df.values[:, 0].astype(int)
        
        # 计算簇数（排除噪声标签-1）
        n_cluster = len(set(y) - {-1})

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        start_time = time.time()

        final_granules = get_gb_division_x(X_scaled)
        
        split_end_time = time.time()
        
        num_granules = len(final_granules)

        granule_labels = merge_granules_with_quantum_proximity(final_granules, n_cluster)
        
        y_pred = np.zeros(len(X), dtype=int)
        granule_idx_to_cluster_label = {i: label for i, label in enumerate(granule_labels)}
        
        for i, gb_obj in enumerate(final_granules):
            final_label = granule_idx_to_cluster_label.get(i, -1)
            for point in gb_obj.data:
                indices = np.where(np.all(np.isclose(X_scaled, point), axis=1))[0]
                if len(indices) > 0:
                    y_pred[indices] = final_label
        
        total_end_time = time.time()
        
        split_duration = split_end_time - start_time
        total_duration = total_end_time - start_time
        
        if 'noise' in data_name or -1 in y:
            valid_mask = (y != -1)
            
            y_true_eval = y[valid_mask]
            y_pred_eval = y_pred[valid_mask]
            
            # 计算指标
            acc, nmi, _, _ = evaluation(y_true_eval, y_pred_eval)
        else:
            # 如果是纯净数据集，直接评估所有点
            acc, nmi, _, _ = evaluation(y, y_pred)
        
        print(data_name_pure)
        print(f"split consume time is----- {format_time_str(split_duration)}")
        print(f"gb_list len: {num_granules}")
        print(f"{n_cluster} acc nmi time: [['{acc:.3f}'], ['{nmi:.3f}'], ['{total_duration:.3f}']]")
        print("--------------------")

if __name__ == '__main__':
    main()