import numpy as base_np
from sklearn.cluster import AgglomerativeClustering


from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector

def merge_granules_with_quantum_proximity(granules, K):
    m = len(granules)
    if m <= K:
        return base_np.arange(m)


    granule_sizes = base_np.array([g.num for g in granules])
    minimum_ball_size = 2 
    size_mask = granule_sizes >= minimum_ball_size

    if base_np.sum(size_mask) < K:
        size_mask = base_np.ones(m, dtype=bool)

    candidate_indices = base_np.where(size_mask)[0]
    m_cand = len(candidate_indices)
    
    if m_cand <= K:
        final_labels = base_np.full(m, -1, dtype=int)
        final_labels[candidate_indices] = base_np.arange(m_cand)
        final_labels[final_labels == -1] = 0
        return final_labels

    # 获取候选粒球的中心向量
    centers = base_np.array([granules[i].center for i in candidate_indices])

    
    def run_swap_test(vec1, vec2):
        """
        构建并运行 Swap Test 线路以计算两个向量的量子保真度。
        """
        n = len(vec1)
        
        # 寄存器定义：
        # anc: 辅助比特 (1个)
        # reg1: 存储 vec1 (n个)
        # reg2: 存储 vec2 (n个)
        anc = QuantumRegister(1, 'ancilla')
        reg1 = QuantumRegister(n, 'vec1')
        reg2 = QuantumRegister(n, 'vec2')
        cr = ClassicalRegister(1, 'c')
        
        qc = QuantumCircuit(anc, reg1, reg2, cr)

        # 1. 编码 State 1 (|psi>) 到 reg1
        for i, feature in enumerate(vec1):
            qc.ry(feature * base_np.pi, reg1[i])
            
        # 2. 编码 State 2 (|phi>) 到 reg2
        for i, feature in enumerate(vec2):
            qc.ry(feature * base_np.pi, reg2[i])

        # 3. Swap Test
        qc.h(anc[0])
        for i in range(n):
            # 受控交换门
            qc.cswap(anc[0], reg1[i], reg2[i])
        qc.h(anc[0])

        sv = Statevector(qc)
        
        probs = sv.probabilities([0])
        p0 = probs[0]
        
        # 计算保真度
        fidelity = 2 * p0 - 1
        return max(0.0, min(1.0, fidelity))

    similarity_matrix = base_np.zeros((m_cand, m_cand))
    
    for i in range(m_cand):
        similarity_matrix[i, i] = 1.0
        for j in range(i + 1, m_cand):
            fid = run_swap_test(centers[i], centers[j])
            similarity_matrix[i, j] = fid
            similarity_matrix[j, i] = fid 

    base_np.fill_diagonal(similarity_matrix, 0) 


    # 凝聚度过滤 
    k_filter = 5 
    if m_cand <= k_filter: k_filter = m_cand - 1
    
    if k_filter > 0:
        sorted_sims = base_np.sort(similarity_matrix, axis=1)
        top_k_sims = sorted_sims[:, -k_filter:]
        cohesion_scores = base_np.mean(top_k_sims, axis=1)
    else:
        cohesion_scores = base_np.zeros(m_cand)

    threshold = base_np.percentile(cohesion_scores, 10) if len(cohesion_scores) > 0 else 0
    core_mask = cohesion_scores >= threshold

    # 核心粒球和潜在噪声粒球
    global_core_indices = candidate_indices[core_mask]
    # global_noise_indices = base_np.concatenate([tiny_noise_indices, candidate_indices[~core_mask]])
    num_cores = len(global_core_indices)

    if num_cores <= K:
        final_labels = base_np.full(m, 0, dtype=int)
        final_labels[global_core_indices] = base_np.arange(num_cores) if num_cores > 0 else 0
    else:
        core_sim_matrix = similarity_matrix[base_np.ix_(core_mask, core_mask)]
        k_neighbors = 5
        if num_cores <= k_neighbors: k_neighbors = num_cores - 1

        # 构建有向邻接矩阵
        directed_adjacency = base_np.zeros((num_cores, num_cores))
        if k_neighbors > 0:
            neighbor_indices = base_np.argsort(core_sim_matrix, axis=1)[:, -k_neighbors:]
            row_indices = base_np.arange(num_cores)[:, base_np.newaxis]
            neighbor_values = core_sim_matrix[row_indices, neighbor_indices]
            directed_adjacency[row_indices, neighbor_indices] = neighbor_values

        # 构建无向邻接矩阵（对称化）
        reinforced_adjacency = (directed_adjacency + directed_adjacency.T) / 2
        # 转为距离
        distance_matrix = 1 - reinforced_adjacency
        base_np.fill_diagonal(distance_matrix, 0)

        # 凝聚式层次聚类
        agg_cluster = AgglomerativeClustering(n_clusters=K, metric='precomputed', linkage='average')
        core_labels = agg_cluster.fit_predict(distance_matrix)

        final_labels = base_np.full(m, -1, dtype=int)
        final_labels[global_core_indices] = core_labels

    
    # 3. 噪声回收
    unassigned_mask = (final_labels == -1)
    unassigned_indices = base_np.where(unassigned_mask)[0]
    assigned_indices = base_np.where(~unassigned_mask)[0]
    
    if len(unassigned_indices) > 0:
        if len(assigned_indices) > 0:
            
            # 遍历每一个噪声球 (unassigned)
            for i, u_idx in enumerate(unassigned_indices):
                best_fid = -1.0
                best_label = -1
                
                # 获取当前噪声球的向量
                noise_vec = granules[u_idx].center
                
                # 遍历所有已分配的核心球 (assigned)
                for a_idx in assigned_indices:
                    # 获取该已分配球的向量
                    assigned_vec = granules[a_idx].center
                    
                    # 执行 Swap Test
                    fid = run_swap_test(noise_vec, assigned_vec)
                    
                    if fid > best_fid:
                        best_fid = fid
                        best_label = final_labels[a_idx]
                
                # 归类
                final_labels[u_idx] = best_label
        else:
            # 如果没有核心球，噪声球归为 0 类
            final_labels[unassigned_indices] = 0

    return final_labels