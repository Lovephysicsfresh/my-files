# 论文非线下实验复现顺序

论文：Active learning guides discovery of a champion four-metal perovskite oxide for oxygen evolution electrocatalysis
DOI: 10.1038/s41563-023-01707-w

## 总原则
严格区分三类内容：
1. 论文明确给出的参数；
2. 作者仓库中已经给出的数据/结果；
3. 论文没有公开、需要我们做复现假设的内容。

## 顺序总览
1. 固定环境和数据版本。
2. 验证30个初始训练样本和40个最终训练样本。
3. 解析PDB图：节点、边、节点特征、键/距离特征、图索引。
4. 复现XRD归一化输入：7个强度 + 7个峰位。
5. 复现MPNN预测模型。
6. 用重复五折交叉验证计算OO-1的预测均值和标准差。
7. 计算Pearson相关系数、RMSE、MAE。
8. 复现XRD生成模型：组成+图 → 14维模拟XRD。
9. 将预测XRD送入全模型，生成候选预测。
10. 生成/读取10,101个候选，确认1,824个不超过四金属候选。
11. 对候选进行图结构随机化和20次/候选或按仓库实现的bootstrap推断。
12. 计算预测均值、不确定度和0.020 V阈值通过率。
13. 对候选特征做t-SNE和K-means，复现补充10个样品的选择逻辑。
14. 将30+10=40个样本合并，重训OO-2。
15. 重复候选推断，检查不确定度改善和CPCF排名。
16. 复现统计检验、特征重要性、特征交互与多相一致性。

## 第1阶段：环境
建议环境：Conda env `oer2022clean`，Python 3.10。
已验证：numpy 1.23.5, pandas 1.5.3, scipy 1.10.1, scikit-learn 1.2.1, matplotlib 3.7.2, tensorflow 2.11.0, keras 2.11.0。

## 第2阶段：数据验收
训练表应满足：
- initial: 30行，final: 40行；
- 10个atom_*列；A位五列之和为1，B位五列之和为1；
- 14个XRD列：intensity_0..6、position_0..6；
- potential无缺失，范围约0.426–0.618 V；
- 候选screen表中potential=0是占位符，不能当作预测结果。

## 第3阶段：公式清单
### 3.1 改进容忍因子
对ABO3：
τ = r_O/r_B − n_A [ n_A − (r_A/r_B) / ln(r_A/r_B) ]
保留 τ < 4.18 的候选。

### 3.2 XRD归一化
I'_j = I_j / max_j(I_j)
θ'_j = (2θ_j) / 90°
七个峰按2θ升序排列；强度和峰位分别形成14维向量。

### 3.3 MPNN消息传递
论文图示给出节点特征4维、键特征3维、pair indices 2维；消息传递层更新节点状态，随后按图聚合为图级向量。
具体层宽、激活和训练超参数必须以SI Supplementary Table 15/作者checkpoint或代码为准，不自行填补。

### 3.4 Bootstrap预测
对每个样本收集10个交叉验证预测值 p_i：
μ_pred = (1/10) Σ_i p_i
σ_pred = sqrt[(1/(10−1)) Σ_i (p_i−μ_pred)^2]
μ_pred是预测过电位，σ_pred是不确定度。

### 3.5 指标
RMSE = sqrt[(1/n)Σ(y_i−ŷ_i)^2]
MAE = (1/n)Σ|y_i−ŷ_i|
R使用实验值与预测均值之间的Pearson相关系数。

### 3.6 预测分布校正
X_corrected = ((X−μ_predicted)/σ_predicted) × σ_experimental + μ_predicted
其中统计量使用30个样本（初始）或40个样本（最终）。

### 3.7 K-means
给定特征向量x_i和k个簇S_j：
argmin_S Σ_j Σ_{x∈S_j} ||x−μ_j||²
μ_j = (1/|S_j|)Σ_{x∈S_j}x
使用训练/候选结构特征，不把潜在标签当聚类输入。

### 3.8 混合熵
ΔS_mix = −R Σ_i x_i ln(x_i)
x_i为各金属摩尔分数；x_i=0项按0处理。

### 3.9 两元素交互
论文SI对元素X、Y的交互使用四种输入情形，按模型输出计算：
δ_XY(S)=f_XY(S)−f_X(S)−f_Y(S)+f_empty(S)
具体空位/占位比例要按SI第5–6节和仓库实现逐项核对。

## 当前第一项任务
先在VS Code中运行：

```powershell
conda activate oer2022clean
cd "E:\组内工作文件\机器学习论文复现\主动学习指导发现冠军四金属钙钛矿析氧电催化剂"
python audit_repository.py
```

必须看到四行PASS。看到四行PASS后，不要直接训练；下一步是检查作者PDB图和补齐作者原始训练脚本/模型定义。
