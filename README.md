# VibCogTriNet

[**简体中文**](./README.md) | [**英语(English)**](./README_EN.md) | [**越南语(Tiếng Việt)**](./README_VN.md)

VibCogTriNet是一个可接受轴承振动波形而进行轴承故障检测的深度学习模型。该模型及其设计方案已用于参加2025年华为杯研究生数学建模大赛，训练该模型所使用的数据集全部由大赛官方提供。

### **数据处理工作流图：**

<img src="./images/data_processing.svg" style="width: 70%;">

### **模型架构图：**

<img src="./images/model_architecture.svg" style="width: 92%;">

### **模型训练损失下降曲线（前42个Epoch为对比学习，后344个Epoch是监督学习）**

<img src="./images/training_loss_curve.svg" style="width: 70%;">

### **模型正向传播梯度范数盒图**

通过梯度归因的方法计算模型内三路特征拼接后输入至全连接分类层的梯度，可以得出这三类特征对模型最后分类效果的“贡献”程度。通过下图可知在测试集上，时域、频域、统计学三路特征在全连接分类器第一层神经网络上的梯度值分布，三路特征在网络上的梯度均大于0，表明它们均对模型最终的分类做出了正面影响。

<img src="./images/gradient_attribution_distribution.svg" style="width: 70%;">


