from Net.basic import *
import torch.nn.functional as F
from Net.ResnetEncoder import ResNetEncoder
import torch
import torch.nn as nn
from Net.kan import KAN
class DenseLayer(torch.nn.Module):
    def __init__(self, in_channels, middle_channels=128, out_channels=32):
        super(DenseLayer, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.BatchNorm1d(in_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(in_channels, middle_channels, 1),
            torch.nn.BatchNorm1d(middle_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(middle_channels, out_channels, 3, padding=1)
        )

    def forward(self, x):
        return torch.cat([x, self.layer(x)], dim=1)
class DenseBlock(torch.nn.Sequential):
    def __init__(self, layer_num, growth_rate, in_channels, middele_channels=128):
        super(DenseBlock, self).__init__()
        for i in range(layer_num):
            layer = DenseLayer(in_channels + i * growth_rate, middele_channels, growth_rate)
            self.add_module('denselayer%d' % (i), layer)

class Transition(torch.nn.Sequential):
    def __init__(self, channels):
        super(Transition, self).__init__()
        self.add_module('norm', torch.nn.BatchNorm1d(channels))
        self.add_module('relu', torch.nn.ReLU(inplace=True))
        self.add_module('conv', torch.nn.Conv1d(channels, channels // 2, 3, padding=1))
        self.add_module('Avgpool', torch.nn.AvgPool1d(2))

class DenseNet(torch.nn.Module):
    def __init__(self, layer_num=(6, 12, 24, 16), growth_rate=32, init_features=64, in_channels=1, middele_channels=128,
                 classes=2):
        """
        1D-DenseNet Module, use to conv global feature and generate final target
        """
        super(DenseNet, self).__init__()
        self.feature_channel_num = init_features
        self.conv = torch.nn.Conv1d(in_channels, self.feature_channel_num, 7, 2, 3)
        self.norm = torch.nn.BatchNorm1d(self.feature_channel_num)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool1d(3, 2, 1)

        self.DenseBlock1 = DenseBlock(layer_num[0], growth_rate, self.feature_channel_num, middele_channels)
        self.feature_channel_num = self.feature_channel_num + layer_num[0] * growth_rate
        self.Transition1 = Transition(self.feature_channel_num)

        self.DenseBlock2 = DenseBlock(layer_num[1], growth_rate, self.feature_channel_num // 2, middele_channels)
        self.feature_channel_num = self.feature_channel_num // 2 + layer_num[1] * growth_rate
        self.Transition2 = Transition(self.feature_channel_num)

        self.DenseBlock3 = DenseBlock(layer_num[2], growth_rate, self.feature_channel_num // 2, middele_channels)
        self.feature_channel_num = self.feature_channel_num // 2 + layer_num[2] * growth_rate
        self.Transition3 = Transition(self.feature_channel_num)

        self.DenseBlock4 = DenseBlock(layer_num[3], growth_rate, self.feature_channel_num // 2, middele_channels)
        self.feature_channel_num = self.feature_channel_num // 2 + layer_num[3] * growth_rate

        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)

        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(self.feature_channel_num, self.feature_channel_num // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(self.feature_channel_num // 2, classes),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.DenseBlock1(x)
        x = self.Transition1(x)

        x = self.DenseBlock2(x)
        x = self.Transition2(x)

        x = self.DenseBlock3(x)
        x = self.Transition3(x)

        x = self.DenseBlock4(x)
        x = self.avgpool(x)
        x = x.view(-1, self.feature_channel_num)
        x = self.classifer(x)
        return x
def cross_concat(tensor1, tensor2):
    batch_size, dim= tensor1.shape

    # 确保两个张量的形状相同
    assert tensor1.shape == tensor2.shape, "两个张量的形状必须相同"

    # 将两个张量在通道维度进行交叉洗牌

    # 交替拼接两个张量的通道
    interleaved = torch.empty(batch_size, dim * 2, device=tensor1.device)  # (B, 1024, D*H*W)
    interleaved[:, 0::2] = tensor1  # 从偶数位置开始填充tensor1
    interleaved[:, 1::2] = tensor2  # 从奇数位置开始填充tensor2
    return interleaved

class CrossModal(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1):
        super(CrossModal, self).__init__()

        # 使用卷积计算 Q, K, V
        self.query_conv = nn.Conv1d(input_channels, output_channels, kernel_size, stride, padding)
        self.key_conv = nn.Conv1d(1, output_channels, kernel_size, stride, padding)
        self.value_conv = nn.Conv1d(1, output_channels, kernel_size, stride, padding)

        self.output_conv = nn.Conv1d(output_channels, output_channels, kernel_size, stride, padding)

        # 添加 BatchNorm1d 归一化层
        self.norm = nn.BatchNorm1d(output_channels)

    def forward(self, image, clinical):
        """
        Args:
            image: [batch_size, channels, seq_len]
            clinical: [batch_size, 1, seq_len]
        Returns:
            context_layer: [batch_size, output_channels, seq_len]
        """

        # 计算 Q, K, V，卷积操作
        query = self.query_conv(image)  # [batch_size, output_channels, seq_len]
        key = self.key_conv(clinical)  # [batch_size, output_channels, seq_len]
        value = self.value_conv(clinical)  # [batch_size, output_channels, seq_len]

        # 计算 Q 和 K 的点积，得到注意力得分
        attention_scores = torch.matmul(query.transpose(1, 2), key)  # [batch_size, seq_len, seq_len]

        # 缩放注意力得分
        attention_scores = attention_scores / math.sqrt(query.size(1))  # 除以 sqrt(d_k)

        # 计算注意力概率（Softmax）
        attention_probs = F.softmax(attention_scores, dim=-1)  # [batch_size, seq_len, seq_len]

        # 对 V 进行加权求和
        context_layer = torch.matmul(attention_probs,
                                     value.transpose(1, 2))  # [batch_size, seq_len, output_channels]

        # 使用卷积进行最终的输出
        context_layer = self.output_conv(
            context_layer.transpose(1, 2))  # [batch_size, output_channels, seq_len]

        # 应用归一化层
        context_layer = self.norm(context_layer)

        return context_layer

class MultiCrossModalInteraction(nn.Module):
    def __init__(self, input_channel_list:list, output_channel_list:list):
        super(MultiCrossModalInteraction, self).__init__()

        self.crossAttn1 = CrossModal(input_channel_list[0], output_channel_list[0], kernel_size=3, stride=1, padding=1)
        self.crossAttn2 = CrossModal(input_channel_list[1], output_channel_list[1], kernel_size=3, stride=1, padding=1)
        self.crossAttn3 = CrossModal(input_channel_list[2], output_channel_list[2], kernel_size=3, stride=1, padding=1)

    def forward(self, Layer1, Layer2, Layer3, Layer4, final_extraction, Cli_input):
        """
        input:
            Layer1: torch.Size([8, 64, 256])
            Layer2: torch.Size([8, 128, 256])
            Layer3: torch.Size([8, 256, 256])
            Layer4: torch.Size([8, 512, 256])
            final_extraction: torch.Size([8, 256])
            Cli_input: torch.Size([8, 1, 256])
        output:
            CMIM_output1: torch.Size([8, 64, 256])
            CMIM_output3: torch.Size([8, 128, 256])
            CMIM_output5: torch.Size([8, 1, 256])
        """
        final_extraction = torch.unsqueeze(final_extraction, dim=1)  # torch.Size([8, 1, 256])
        CMIM_output1 = self.crossAttn1(Layer1, Cli_input)
        # CMIM_output2 = self.crossAttn2(Layer2, Cli_input)
        CMIM_output3 = self.crossAttn2(Layer3, Cli_input)
        # CMIM_output4 = self.crossAttn4(Layer4, Cli_input)
        CMIM_output5 = self.crossAttn3(final_extraction, Cli_input)
        # final_extraction = torch.cat([CMIM_output1, CMIM_output3, final_extraction], dim=1)



        return CMIM_output1,  CMIM_output3, CMIM_output5

class MultiScaleFusionModule(nn.Module):
    def __init__(self):
        super(MultiScaleFusionModule, self).__init__()
    def forward(self, CMIM_input1, CMIM_input2, CMIM_input3):
        output = torch.cat([CMIM_input1, CMIM_input2, CMIM_input3], dim=1)
        return output

class SpecificMultiLayerFusion(nn.Module):
    def __init__(self):
        super(SpecificMultiLayerFusion, self).__init__()
        self.name = 'MultiLayerFusion_ver1.0'
        self.MRI_encoder = ResNetEncoder(BasicBlock, [2, 2, 2, 2], get_inplanes())
        self.PET_encoder = ResNetEncoder(BasicBlock, [2, 2, 2, 2], get_inplanes())
        self.CLI_encoder = TransformerEncoder(output_dim=256)
        self.MRI_CMIM = MultiCrossModalInteraction(input_channel_list=[64, 256, 1], output_channel_list=[64, 256, 1])
        self.PET_CMIM = MultiCrossModalInteraction(input_channel_list=[64, 256, 1], output_channel_list=[64, 256, 1])
        self.MSFM = MultiScaleFusionModule()
        self.mri1_fit = nn.Linear(36864, 256)  # torch.Size([8, 64, 256])
        self.mri2_fit = nn.Linear(4608, 256)  # torch.Size([8, 128, 256])
        self.mri3_fit = nn.Linear(576, 256)  # torch.Size([8, 256, 256])
        self.mri4_fit = nn.Linear(72, 256)  # torch.Size([8, 512, 256])
        self.mrif_fit = nn.Linear(400, 256)  # torch.Size([8, 256])

        self.pet1_fit = nn.Linear(36864, 256)  # torch.Size([8, 64, 256])
        self.pet2_fit = nn.Linear(4608, 256)  # torch.Size([8, 128, 256])
        self.pet3_fit = nn.Linear(576, 256)  # torch.Size([8, 256, 256])
        self.pet4_fit = nn.Linear(72, 256)  # torch.Size([8, 512, 256])
        self.petf_fit = nn.Linear(400, 256)  # torch.Size([8, 256])
        # self.SA1 = SelfAttention(16, 256, 256, hidden_dropout_prob=0.2)
        # self.SA2 = SelfAttention(16, 256, 256, hidden_dropout_prob=0.2)
        # self.SA3 = SelfAttention(16, 256, 256, hidden_dropout_prob=0.2)
        # self.classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=16, in_channels=1, classes=2)

    def forward(self, mri, pet, cli):
        """
        Layer1 torch.Size([8, 64, 48, 32, 24])
        Layer2 torch.Size([8, 128, 24, 16, 12])
        Layer3 torch.Size([8, 256, 12, 8, 6])
        Layer4 torch.Size([8, 512, 6, 4, 3])
        final_extraction torch.Size([8, 400])
        """
        layer1_mri, layer2_mri, layer3_mri, layer4_mri, output_mri = self.MRI_encoder(mri)
        layer1_pet, layer2_pet, layer3_pet, layer4_pet, output_pet = self.PET_encoder(pet)
        cli_feature = self.CLI_encoder(cli)
        cli_feature = torch.unsqueeze(cli_feature, dim=1)  # torch.Size([8, 1, 256])
        # cli_feature = cli_feature.transpose(-1, -2)
        mri_Layer1_flattened = layer1_mri.view(layer1_mri.size(0), layer1_mri.size(1), -1) # torch.Size([8, 64, 36864])
        mri_Layer2_flattened = layer2_mri.view(layer2_mri.size(0), layer2_mri.size(1), -1) # torch.Size([8, 128, 4608])
        mri_Layer3_flattened = layer3_mri.view(layer3_mri.size(0), layer3_mri.size(1), -1) # torch.Size([8, 256, 576])
        mri_Layer4_flattened = layer4_mri.view(layer4_mri.size(0), layer4_mri.size(1), -1) # torch.Size([8, 512, 72])


        MRI_CMIM_output1, MRI_CMIM_output2, MRI_CMIM_output3 = self.MRI_CMIM(self.mri1_fit(mri_Layer1_flattened),
                                                                             self.mri2_fit(mri_Layer2_flattened),
                                                                             self.mri3_fit(mri_Layer3_flattened),
                                                                             self.mri4_fit(mri_Layer4_flattened),
                                                                             self.mrif_fit(output_mri), cli_feature)


        pet_Layer1_flattened = layer1_pet.view(layer1_pet.size(0), layer1_pet.size(1), -1)
        pet_Layer2_flattened = layer2_pet.view(layer2_pet.size(0), layer2_pet.size(1), -1)
        pet_Layer3_flattened = layer3_pet.view(layer3_pet.size(0), layer3_pet.size(1), -1)
        pet_Layer4_flattened = layer4_pet.view(layer4_pet.size(0), layer4_pet.size(1), -1)


        PET_CMIM_output1, PET_CMIM_output2, PET_CMIM_output3 = self.PET_CMIM(self.pet1_fit(pet_Layer1_flattened),
                                                                             self.pet2_fit(pet_Layer2_flattened),
                                                                             self.pet3_fit(pet_Layer3_flattened),
                                                                             self.pet4_fit(pet_Layer4_flattened),
                                                                             self.petf_fit(output_pet), cli_feature)
        mri_output = self.MSFM(MRI_CMIM_output1, MRI_CMIM_output2, MRI_CMIM_output3)
        pet_output = self.MSFM(PET_CMIM_output1, PET_CMIM_output2, PET_CMIM_output3)

        return mri_output, pet_output

class SharedMultiScaleExtraction(nn.Module):
    def __init__(self):
        super(SharedMultiScaleExtraction, self).__init__()
        self.name = 'SharedKMultiScaleFeatureExtractionLayer'
        self.sharedFeatureExtractor = ResNetEncoder(BasicBlock, [2, 2, 2, 2], get_inplanes())
    def forward(self, mri, pet, cli):
        layer1_mri, layer2_mri, layer3_mri, layer4_mri, output_mri = self.sharedFeatureExtractor(mri)
        layer1_pet, layer2_pet, layer3_pet, layer4_pet, output_pet = self.sharedFeatureExtractor(pet)
        output = cross_concat(output_mri, output_pet)
        return output

class TriModalCrossAttention_ver2(nn.Module):
    def __init__(self, input_dim):
        super(TriModalCrossAttention_ver2, self).__init__()

        # 定义 Query, Key, Value 的线性变换
        self.W_q = nn.Linear(input_dim * 3, input_dim)  # 用于拼接后的查询
        self.W_k1 = nn.Linear(input_dim, input_dim)
        self.W_v1 = nn.Linear(input_dim, input_dim)

        self.W_k2 = nn.Linear(input_dim, input_dim)
        self.W_v2 = nn.Linear(input_dim, input_dim)

        self.W_k3 = nn.Linear(input_dim, input_dim)
        self.W_v3 = nn.Linear(input_dim, input_dim)

        # 输出投影
        self.W_o1 = nn.Linear(input_dim * 2, input_dim)
        self.W_o2 = nn.Linear(input_dim * 2, input_dim)
        self.W_o3 = nn.Linear(input_dim * 2, input_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x1, x2, x3):
        # x1, x2, x3: [B, N, input_dim] [8, 256, 1]
        batch_size, seq_len, input_dim = x1.size()

        # 将三个模态在最后一个维度拼接得到 [B, N, input_dim * 3]
        combined_input = torch.cat([x1, x2, x3], dim=-1)
        queries = self.W_q(combined_input)  # [B, N, input_dim * 3]

        # 对每个模态的键和值进行投影
        keys1 = self.W_k1(x1)  # [B, N, input_dim]
        values1 = self.W_v1(x1)

        keys2 = self.W_k2(x2)  # [B, N, input_dim]
        values2 = self.W_v2(x2)

        keys3 = self.W_k3(x3)  # [B, N, input_dim]
        values3 = self.W_v3(x3)

        # 分别计算每个模态的交叉注意力输出
        attention_scores1 = torch.matmul(queries, keys1.transpose(-2, -1)) / (input_dim ** 0.5)  # [B, N, N]
        attention_weights1 = F.softmax(attention_scores1, dim=-1)
        context1 = torch.matmul(self.dropout(attention_weights1), values1)  # [B, N, input_dim]

        attention_scores2 = torch.matmul(queries, keys2.transpose(-2, -1)) / (input_dim ** 0.5)  # [B, N, N]
        attention_weights2 = F.softmax(attention_scores2, dim=-1)
        context2 = torch.matmul(self.dropout(attention_weights2), values2)  # [B, N, input_dim]

        attention_scores3 = torch.matmul(queries, keys3.transpose(-2, -1)) / (input_dim ** 0.5)  # [B, N, N]
        attention_weights3 = F.softmax(attention_scores3, dim=-1)
        context3 = torch.matmul(self.dropout(attention_weights3), values3)  # [B, N, input_dim]

        # 将原输入与注意力上下文拼接
        combined1 = torch.cat((x1, context1), dim=-1)  # [B, N, input_dim * 2]
        combined2 = torch.cat((x2, context2), dim=-1)  # [B, N, input_dim * 2]
        combined3 = torch.cat((x3, context3), dim=-1)  # [B, N, input_dim * 2]

        # 线性变换生成最终输出
        output1 = self.W_o1(combined1)  # [B, N, input_dim]
        output2 = self.W_o2(combined2)  # [B, N, input_dim]
        output3 = self.W_o3(combined3)  # [B, N, input_dim]

        # 将三个模态的输出在第二维度拼接 [B, N * 3, input_dim]
        vision_feature = shuffle_interleave(output1, output2)
        # global_feature = torch.cat((output1, output2, output3), dim=1)
        global_feature = torch.cat((vision_feature,output3), dim=1)

        return output1, output2, output3, global_feature


class MultiLayerFusionModel(nn.Module):
    def __init__(self):
        super(MultiLayerFusionModel, self).__init__()
        self.sharedFeatureExtractor = SharedMultiScaleExtraction()
        self.specificFeatureFusion = SpecificMultiLayerFusion()
        self.triModalAttention = TriModalCrossAttention_ver2(input_dim=1)
        self.classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=16, in_channels=1, classes=2)
        # 使用全局平均池化将321维压缩到1维
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.image_fit_model = nn.Linear(321, 256)
        # self.image_fit_model2 = nn.Linear(256, 1)
        self.shared_fit_model = KAN([800, 256])
    def forward(self, mri, pet, cli):
        """
         specific_mri_output torch.Size([8, 321, 256])
         specific_pet_output torch.Size([8, 321, 256])
         shared_mri_pet_output torch.Size([8, 800])
        """
        specific_mri_output, specific_pet_output = self.specificFeatureFusion(mri, pet, cli)
        shared_mri_pet_output = self.sharedFeatureExtractor(mri, pet, cli)

        specific_mri_output = self.pool(specific_mri_output)
        specific_mri_output = specific_mri_output.squeeze(dim=-1)
        specific_mri_output_fit = self.image_fit_model(specific_mri_output).unsqueeze(dim=-1)
        specific_pet_output = self.pool(specific_pet_output)
        specific_pet_output = specific_pet_output.squeeze(dim=-1)
        specific_pet_output_fit = self.image_fit_model(specific_pet_output).unsqueeze(dim=-1)

        shared_mri_pet_output_fit = self.shared_fit_model(shared_mri_pet_output).unsqueeze(dim=-1)
        #print("specific_mri_output_fit", specific_mri_output_fit.shape)
        #print("specific_pet_output_fit", specific_pet_output_fit.shape)
        #print("shared_mri_pet_output_fit", shared_mri_pet_output_fit.shape)

        output1, output2, output3, global_feature = self.triModalAttention(specific_mri_output_fit, specific_pet_output_fit, shared_mri_pet_output_fit)
        global_feature = global_feature.permute(0, 2, 1)
        classified_output = self.classify_head(global_feature)
        return output1, output2, output3, classified_output

# if __name__ == '__main__':
#     x = torch.randn(8, 1, 96, 128, 96)
#     y = torch.randn(8, 1, 96, 128, 96)
#     z = torch.randn(8, 9)
#     model = MultiLayerFusionModel()
#     print("参数量 for model：", sum(p.numel() for p in model.parameters()))
#     print(model(x, y, z).shape)
    # model = SpecificMultiLayerFusion()
    # model2 = SharedMultiScaleExtraction()
    # print("参数量 for model：", sum(p.numel() for p in model.parameters()))
    # print("参数量 for model2：", sum(p.numel() for p in model2.parameters()))
    # mri_output, pet_output = model(x, y, z)
    # shared_output = model2(x, y, z)
    # print("mri_output", mri_output.shape)
    # print("pet_output", pet_output.shape)
    # print("shared_output", shared_output.shape)
    """
    mri_output torch.Size([8, 321, 256])
    pet_output torch.Size([8, 321, 256])
    shared_output torch.Size([8, 800])
    """
    # a = torch.randn([8, 256, 1])
    # b = torch.randn([8, 256, 1])
    # c = torch.randn([8, 256, 1])
    # attention_model = TriModalCrossAttention_ver2(input_dim=1)
    # output1, output2, output3, global_feature = attention_model(a, b, c)
    # print("参数量 for attention_model", sum(p.numel() for p in attention_model.parameters()))
    # print(output1.shape)
    # print(output2.shape)
    # print(output3.shape)
    # print(global_feature.shape)
    # classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=16, in_channels=1, classes=2)
    # print("参数量 for classify_head", sum(p.numel() for p in classify_head.parameters()))
    # print(classify_head(global_feature.permute(0, 2, 1)).shape)
