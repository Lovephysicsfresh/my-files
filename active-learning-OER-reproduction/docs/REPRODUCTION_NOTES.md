# OER2022 复现项目

## 当前来源
- 论文正文：2.Active learning guides discovery of a champion four-metal perovskite oxide for oxygen evolution electrocatalysis.pdf
- Supporting Information：2.SI Active learning guides discovery of a champion four-metal perovskite oxide for oxygen evolution electrocatalysis.pdf
- 作者仓库：https://github.com/JunseokMoon/OER2022
- DOI：10.1038/s41563-023-01707-w

## 复现边界
本项目先复现非线下实验部分：数据读取、图结构输入、MPNN预测、bootstrap集成、候选筛选、t-SNE/K-means和统计分析；不执行材料合成、XRD测量或电化学实验。

## 已核对的关键参数
- 初始数据：30个氧化物；最终数据：40个氧化物。
- 图：2×2×2 Pnma超胞；约336个节点。
- 金属节点特征：离子半径、Pauling电负性、A/B位one-hot。
- XRD：7个归一化强度 + 7个归一化峰位。
- 初始候选：10,101个；其中不超过四金属1,824个。
- 初始OO-1：相关系数约0.882，RMSE约0.033 V。
- 最终OO-2：相关系数约0.897，RMSE约0.025 V。
- 不确定度阈值：0.020 V。

## 当前状态
作者仓库的README及核心CSV已放入 `OER2022/`。完整GitHub压缩包下载受网络/传输限制影响未完整解压，因此后续优先使用已验证的核心数据，并按需下载单个脚本或图文件。