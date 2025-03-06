import torch
from torch.utils.data import DataLoader
from Dataset import MriPetDatasetNew
from torchvision.transforms.functional import to_pil_image
from Net.MultiLayerFusion import MultiLayerFusionModelWithLatentFusion
from Net.TripleNetwork import Interactive_Multimodal_Fusion_Model
import numpy as np
from PIL import Image  # 处理高质量图片
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


if __name__ == '__main__':
    torch.manual_seed(42)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # 加载模型
    pretrainedModelPath = r'/data3/wangchangmiao/shenxy/Code/MICCAI/TripleNet_IMF/logs/classification/Interactive Multimodal Fusion Model_2025-02-23_12-59_fold_4/logs/best_model.pth'  # 替换为你的模型路径
    model = Interactive_Multimodal_Fusion_Model()
    model.load_state_dict(torch.load(pretrainedModelPath, map_location=device))
    model = model.to(device)
    model.eval()

    # 加载数据
    mri_dir = r'/data3/wangchangmiao/shenxy/ADNI/ADNI2/MRI'
    pet_dir = r'/data3/wangchangmiao/shenxy/ADNI/ADNI2/PET'
    cli_dir = r'./ADNI_Clinical.csv'
    csv_file = r'./ADNI2_test.csv'
    batch_size = 8
    dataset = MriPetDatasetNew(mri_dir, pet_dir, cli_dir, csv_file, valid_group=("pMCI", "sMCI"))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 假设你的标签是二分类问题，并且存储在label中
    all_labels = []
    all_probabilities = []

    # 在循环外部初始化列表以保存所有预测概率和真实标签

    with torch.no_grad():  # 关闭梯度计算
        for batch_idx, (mri, pet, clinical, label, label_2d) in enumerate(data_loader):
            mri, pet = mri.to(device), pet.to(device)
            clinical = clinical.to(device)

            # 前向传播
            outputs = model(mri, pet, clinical)
            probabilities = (outputs[0] + outputs[1] + outputs[2] + outputs[3]) / 4.0

            # 保存当前批次的概率和标签
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # 假设正类是索引为1的列
            all_labels.extend(label.cpu().numpy())

    # 计算ROC曲线的相关指标
    print("all_labels",all_labels)
    print("all_probabilities", all_probabilities)
    fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.figure()
    lw = 2  # 线宽
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig('roc_curve.png')




"""
已知这是写好的代码，输入数据有三个，MRI PET 和表格数据CLI，
输出为概率【batch, 2】，请基于以下代码继续写
"""
    # for i in range(len(dataset)):
    #     mri_img, pet_img, cli_tab, _, _ = dataset[i]
    #     print(f"{i} MRI Image shape: {mri_img.shape}")  # ([1, 96, 128, 96])
    #     print(f"{i} PET Image shape: {pet_img.shape}")  # ([1, 96, 128, 96])
    #     print(f"{i} Clinical Table shape: {cli_tab.shape}")  #([9])
# def mark_cli_data(cli_data, predicted_classes):
#     """Mark red on CLI table based on predicted class."""
#     marked_cli = cli_data.copy()
#     # Assuming CLI data contains columns such as "Age", "Sex", etc.
#     # You can add more logic to mark specific columns
#     for i, pred_class in enumerate(predicted_classes):
#         if pred_class == 1:  # Assuming 1 means AD, for example
#             marked_cli.iloc[i] = marked_cli.iloc[i].apply(
#                 lambda x: f"\033[91m{x}\033[0m" if isinstance(x, str) else x)  # Mark in red
#     return marked_cli
#
#
# def save_cli_data(cli_data, batch_idx):
#     """Save the CLI data (as CSV for example) for the current batch."""
#     cli_data.to_csv(f"marked_cli_batch_{batch_idx}.csv", index=False)

"""
请实现抓取梯度，生成热力图
已知：图片是3D的，模型是三模态的，dataset和模型已经定义好，每个数据都有mri pet clinical(临床数据)，
现在请你用python实现，每次同时输入三模态数据 【mri pet clinical】，
然后分别为mri pet 的切片生成热力图，请用现有的包实现，如torchcam grad-cam
"""