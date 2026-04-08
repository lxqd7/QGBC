import time
import numpy as np
from sklearn.cluster import kmeans_plusplus
from collections import deque  

class GranularBall:
    def __init__(self, data):
        self.data = data
        self.num = len(data)
        
        if self.num > 0:
            self.center = np.mean(data, axis=0)
            
            # 【微调优化】：不再重复开根号，保存距离向量和最大半径
            # np.linalg.norm(..., axis=1) 底层也是这样算的，我们提取出来备用
            diff = data - self.center
            self.sq_dists = np.sum(diff ** 2, axis=1)
            self.radius = np.sqrt(np.max(self.sq_dists))
        else:
            self.center = None
            self.radius = 0.0
            self.sq_dists = None
            
    def __len__(self):
        return self.num

    def __repr__(self):
        return f"<GB: num={self.num}, r={self.radius:.4f}>"


# ================================================================= #
#                阶段一：保持原样 (初始划分)                          #
# ================================================================= #
def fast_probabilistic_seeding_partition(X, k):
    n_samples = X.shape[0]
    if k <= 1 or n_samples <= k: 
        return[GranularBall(X)]

    centers = [X[np.random.randint(n_samples)]]
    dist_sq = np.linalg.norm(X - centers[0], axis=1)**2

    l = 2 * k 
    n_local_trials = 5
    for _ in range(n_local_trials):
        potential = dist_sq.sum()
        if potential == 0: break
        probs = dist_sq / potential
        new_indices = np.random.choice(n_samples, size=l, p=probs, replace=True)
        centers.extend(X[new_indices])
        
        last_added_centers = np.array(centers[-l:])
        new_dist_sq_batch = np.min(np.linalg.norm(X[:, np.newaxis] - last_added_centers, axis=2)**2, axis=1)
        dist_sq = np.minimum(dist_sq, new_dist_sq_batch)

    candidate_centers = np.array(centers)
    final_centers, _ = kmeans_plusplus(candidate_centers, n_clusters=k, random_state=5)

    labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - final_centers, axis=2), axis=1)
    gb_list =[]
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            gb_list.append(GranularBall(cluster_points))
            
    return gb_list


# ================================================================= #
#       阶段二：【学术+工程优化】使用 Quickselect 替代排序寻找 Core     #
# ================================================================= #
def split_by_robust_pca(gb):
    """
    保证完全相同的数值结果，但剔除了 O(N log N) 的排序开销。
    """
    data = gb.data 
    n_samples = gb.num
    
    CORE_PERCENTAGE = 0.9
    MIN_SPLIT_SIZE = 4
    
    if n_samples < MIN_SPLIT_SIZE: 
        return[]

    try:
        # 直接使用我们在初始化时计算好的平方距离，避免重新算一遍欧式距离！
        distances_to_center_sq = gb.sq_dists
        
        core_size = int(n_samples * CORE_PERCENTAGE)
        if core_size < 2: 
            core_size = n_samples
            
        if core_size >= n_samples:
            clean_core = data
        else:
            # ---------------------------------------------------------
            # 【核心优化】：使用 np.argpartition 实现 O(N) 复杂度的核心集过滤
            # 原代码 np.percentile 内部会调用完全排序 O(N log N)。
            # np.argpartition 基于 Quickselect 算法，只需找到分割点，速度极快！
            # 并且我们直接比较“平方距离”，省去了 N 次昂贵的开根号(sqrt)运算。
            # ---------------------------------------------------------
            idx = np.argpartition(distances_to_center_sq, core_size - 1)
            core_indices = idx[:core_size]
            clean_core = data[core_indices]

        # 恢复完全一致的 PCA 计算（对低维数据这是最快的，且结果精确）
        core_center = np.mean(clean_core, axis=0)
        core_centered = clean_core - core_center
        covariance_matrix = np.dot(core_centered.T, core_centered)
        _, eigenvectors = np.linalg.eigh(covariance_matrix)
        principal_component = eigenvectors[:, -1]

        data_centered = data - gb.center
        projections = np.dot(data_centered, principal_component)
        mask_ball_1 = projections < 0
        
        data_1 = data[mask_ball_1]
        data_2 = data[~mask_ball_1]
        
        if len(data_1) == 0 or len(data_2) == 0:
            return []
            
        return[GranularBall(data_1), GranularBall(data_2)]
        
    except np.linalg.LinAlgError:
        return[]

def calculate_sse(gb):
    if gb.num == 0: return 0.0
    # 直接复用粒球自带的平方距离，复杂度 O(1)，取代原来的实时重算 O(Nd)
    return np.sum(gb.sq_dists)

def calculate_split_benefit(gb):
    if gb.num < 4:
        return -1, None, None
        
    children_gbs = split_by_robust_pca(gb)
    if not children_gbs:
        return -1, None, None
        
    gb1, gb2 = children_gbs[0], children_gbs[1]
    
    # 因为 SSE 的计算逻辑已优化，这里获取 SSE 极其迅速
    sse_parent = calculate_sse(gb)
    sse_children = calculate_sse(gb1) + calculate_sse(gb2)

    benefit = sse_parent - sse_children

    if benefit > 0:
        return benefit, gb1, gb2
    else:
        return -1, None, None


def get_gb_division_x(data, plt_flag=False):
    k1 = max(1, int(np.sqrt(len(data))))
    RELATIVE_BENEFIT_THRESHOLD = 0.01 

    initial_granules = fast_probabilistic_seeding_partition(data, k1)

    max_initial_benefit = 0.0
    initial_splits =[]

    for gb in initial_granules:
        benefit, child1, child2 = calculate_split_benefit(gb)
        if benefit > 0:
            if benefit > max_initial_benefit:
                max_initial_benefit = benefit
            initial_splits.append((benefit, gb, child1, child2))
        else:
            initial_splits.append((-1, gb, None, None))

    if max_initial_benefit == 0:
        return initial_granules

    stop_threshold = max_initial_benefit * RELATIVE_BENEFIT_THRESHOLD

    processing_queue = deque()
    final_granules =[]

    for benefit, gb, child1, child2 in initial_splits:
        if benefit > stop_threshold:
            processing_queue.append(child1)
            processing_queue.append(child2)
        else:
            final_granules.append(gb)
            
    while processing_queue:
        current_gb = processing_queue.popleft()
        
        if current_gb.num > 0:
            new_benefit, new_child1, new_child2 = calculate_split_benefit(current_gb)
            
            if new_benefit > stop_threshold:
                processing_queue.append(new_child1)
                processing_queue.append(new_child2)
            else:
                final_granules.append(current_gb)
    
    return [g for g in final_granules if g.num > 0]