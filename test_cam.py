import torch
from skimage import transform as skt
from torch.utils.data import DataLoader
from Dataset import MriPetDatasetNew
from torchcam.methods import SmoothGradCAMpp
from torchvision.transforms.functional import to_pil_image
from torchcam.utils import overlay_mask
from Net.MultiLayerFusion import MultiLayerFusionModelWithLatentFusion
import numpy as np
from PIL import Image  # 处理高质量图片
import matplotlib.pyplot as plt
class CustomSmoothGradCAMpp(SmoothGradCAMpp):
    def __init__(self, model, target_layer, extra_args=None):
        super().__init__(model, target_layer)
        self.extra_args = extra_args if extra_args is not None else {}

    def forward(self, *input_tensor):
        return self.model(*input_tensor, **self.extra_args)


if __name__ == '__main__':
    torch.manual_seed(42)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # 加载模型
    pretrainedModelPath = r'/data3/wangchangmiao/shenxy/Code/MICCAI/MLF_new_adni1/logs/classification/MLFLatentFusion_2025-02-24_20-48_fold_4/logs/best_model.pth'  # 替换为你的模型路径
    model = MultiLayerFusionModelWithLatentFusion()
    model.load_state_dict(torch.load(pretrainedModelPath, map_location=device))
    model = model.to(device)
    model.eval()

    # 加载数据
    mri_dir = r'/data3/wangchangmiao/shenxy/ADNI/ADNI1/MRI'
    pet_dir = r'/data3/wangchangmiao/shenxy/ADNI/ADNI1/PET'
    cli_dir = r'./ADNI_Clinical.csv'
    csv_file = r'./ADNI1_match.csv'
    batch_size = 8
    dataset = MriPetDatasetNew(mri_dir, pet_dir, cli_dir, csv_file, valid_group=("pMCI", "sMCI"))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 选择目标层
    # target_layer = model.triModalFusion.modal3_layers
    target_layer = model.specificFeatureFusion.MRI_encoder.layer4[-1]

    # 初始化Grad-CAM方法
    # cam_extractor = SmoothGradCAMpp(model, target_layer)




    for batch_idx, (mri, pet, clinical, _, _) in enumerate(data_loader):
        # 使用自定义的 CAM 提取器
        cam_extractor = CustomSmoothGradCAMpp(model, target_layer, {'pet':pet, 'clinical': clinical})
        mri, pet = mri.to(device), pet.to(device)
        clinical = clinical.to(device)

        # 前向传播
        outputs = model(mri, pet, clinical)
        # _, predicted = torch.max(outputs, 1)

        probabilities = (outputs[0] + outputs[1] + outputs[2] + outputs[3]) / 4.0
        _, predicted = torch.max(probabilities, dim=1)

        # 计算热力图
        for idx in range(batch_size):
            # 获取单个样本的输出
            output = probabilities[idx].unsqueeze(0)
            predicted_class = predicted[idx].item()
            print("output", output.shape)
            print("predicted_class", predicted_class)

            # 计算热力图
            # activation_map = cam_extractor(predicted_class, output)[0].cpu().numpy()

            # 修改计算热力图的部分
            activation_map = cam_extractor(predicted_class, output)[0].cpu().numpy()

            # 叠加热力图到原始图像
            if idx < len(mri):  # 确保索引在范围内
                mri_img = to_pil_image(mri[idx].cpu())
                pet_img = to_pil_image(pet[idx].cpu())

                # 调整大小以匹配原始图像
                activation_map_resized = skt.resize(activation_map, mri_img.size, anti_aliasing=True)

                # 叠加热力图
                mri_heatmap = overlay_mask(mri_img, to_pil_image(torch.from_numpy(activation_map_resized), mode='F'),
                                           alpha=0.5)
                pet_heatmap = overlay_mask(pet_img, to_pil_image(torch.from_numpy(activation_map_resized), mode='F'),
                                           alpha=0.5)

                # 显示和保存结果
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.title('MRI Heatmap')
                plt.imshow(mri_heatmap)
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.title('PET Heatmap')
                plt.imshow(pet_heatmap)
                plt.axis('off')

                plt.tight_layout()
                plt.show()

                # 保存结果
                mri_heatmap.save(f'mri_heatmap_batch{batch_idx}_idx{idx}.png')
                pet_heatmap.save(f'pet_heatmap_batch{batch_idx}_idx{idx}.png')
        if batch_idx == 1:
            break



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