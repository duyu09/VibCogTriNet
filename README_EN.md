# VibCogTriNet

[**Chinese Traditional(繁體中文)**](./README.md) | [**English**](./README_EN.md) | [**Vietnamese(Tiếng Việt)**](./README_VN.md)

VibCogTriNet is a deep learning model designed for bearing fault detection based on vibration waveforms. The model integrates **time-domain**, **frequency-domain**, and **statistical features** of bearing vibration signals, and makes a comprehensive judgment on the type of fault. This model and its design scheme were used in the **2025 Huawei Cup Graduate Mathematical Modeling Contest**. All datasets used for model training were officially provided by the competition organizers.

---

### **Data Processing Workflow:**

<img src="./images/data_processing.svg" style="width: 70%;">

---

### **Model Architecture:**

<img src="./images/model_architecture.svg" style="width: 92%;">

---

### **Training Loss Curve (First 42 Epochs: Contrastive Learning; Following 344 Epochs: Supervised Learning)**

<img src="./images/training_loss_curve.svg" style="width: 70%;">

---

### **Forward Propagation Gradient Norm Boxplot**

By applying gradient attribution methods, the gradients of the concatenated three-way features fed into the fully connected classifier are calculated, which reflects the degree of “contribution” of each feature type to the final classification result. The figure below shows the distribution of gradient values of time-domain, frequency-domain, and statistical features on the first layer of the fully connected classifier in the test set. Since all three feature gradients are greater than 0, it indicates that they all make positive contributions to the final classification of the model.

<img src="./images/gradient_attribution_distribution.svg" style="width: 70%;">

---

### **Ablation Study Results:**

| Model                          | Accuracy     | F1-score     |
| ------------------------------ | ------------ | ------------ |
| ***VibcogTriNet***             | ***0.9969*** | ***0.9538*** |
| VibcogTriNet-no-Transformer    | 0.9812       | 0.9501       |
| VibcogTriNet-no-SpectrogramCNN | 0.9289       | 0.9001       |
| VibcogTriNet-no-Stats          | 0.9794       | 0.9444       |

---

### **Project Authors**

**Copyright © 2025** [**HE Feifan**](https://faculty.lzjtu.edu.cn/chenmei/zh_CN/xsxx/2554/content/1835.htm) (Chinese: 何非凡; Vietnamese: HÀ Phi Phàm), [**DU Yu**](https://faculty.lzjtu.edu.cn/chenmei/zh_CN/xsxx/2554/content/1837.htm) (Chinese: 杜宇; Vietnamese: ĐỖ Vũ), [**YANG Shasha**](https://faculty.lzjtu.edu.cn/chenmei/zh_CN/xsxx/2554/content/1836.htm) (Chinese: 楊莎莎; Vietnamese: DƯƠNG Sa Sa), [**School of Electronic and Information Engineering, Lanzhou Jiaotong University**](https://dxxy.lzjtu.edu.cn/) (Chinese: 蘭州交通大學電子與信息工程學院; Vietnamese: Đại Học Giao thông Lan Châu, Học Viện Điện Tử Và Công Nghệ Thông Tin)

