import numpy as np
from text_process import get_details
from text_process import get_details1
from text_process import get_max_evaluation_index
import image_visualization

# arr_lr2, arr_epoch2, arr_all_loss2, arr_m_loss2, arr_c_loss2, arr_s_loss2, arr_i_loss2,  \
#         precision_iCoseg2, precision_MSRC72, precision_int3002, precision_PAS2, jaccard_iCoseg2, \
#         jaccard_MSRC72, jaccard_int3002, jaccard_PAS2 \
#         = get_details("datas/train_bs2.log")

arr_lr2, arr_epoch2, arr_all_loss2, arr_m_loss2, arr_c_loss2, arr_s_loss2, arr_i_loss2, arr_sfm_loss2,  \
        precision_iCoseg2, precision_MSRC72, precision_int3002, precision_PAS2, jaccard_iCoseg2, \
        jaccard_MSRC72, jaccard_int3002, jaccard_PAS2 \
        = get_details1("datas/train2sfm.log")

# arr_lr4, arr_epoch4, arr_all_loss4, arr_m_loss4, arr_c_loss4, arr_s_loss4, arr_i_loss4,  \
#         precision_iCoseg4, precision_MSRC74, precision_int3004, precision_PAS4, jaccard_iCoseg4,\
#         jaccard_MSRC74, jaccard_int3004, jaccard_PAS4 \
#         = get_details("datas/train_bs4.log")

arr_lr4, arr_epoch4, arr_all_loss4, arr_m_loss4, arr_c_loss4, arr_s_loss4, arr_i_loss4,arr_sfm_loss4,  \
        precision_iCoseg4, precision_MSRC74, precision_int3004, precision_PAS4, jaccard_iCoseg4,\
        jaccard_MSRC74, jaccard_int3004, jaccard_PAS4 \
        = get_details1("datas/train4mse.log")


# arr_lr8, arr_epoch8, arr_all_loss8, arr_m_loss8, arr_c_loss8, arr_s_loss8, arr_i_loss8,  \
#         precision_iCoseg8, precision_MSRC78, precision_int3008, precision_PAS8, jaccard_iCoseg8,\
#         jaccard_MSRC78, jaccard_int3008, jaccard_PAS8 \
#         = get_details("datas/train_bs8.log")

arr_lr8, arr_epoch8, arr_all_loss8, arr_m_loss8, arr_c_loss8, arr_s_loss8, arr_i_loss8, arr_sfm_loss8, \
        precision_iCoseg8, precision_MSRC78, precision_int3008, precision_PAS8, jaccard_iCoseg8,\
        jaccard_MSRC78, jaccard_int3008, jaccard_PAS8 \
        = get_details1("datas/train8mse.log")

# arr_lr16, arr_epoch16, arr_all_loss16, arr_m_loss16, arr_c_loss16, arr_s_loss16, arr_i_loss16,  \
#         precision_iCoseg16, precision_MSRC716, precision_int30016, precision_PAS16, jaccard_iCoseg16,\
#         jaccard_MSRC716, jaccard_int30016, jaccard_PAS16 \
#         = get_details("datas/train_bs16.log")

arr_lr16, arr_epoch16, arr_all_loss16, arr_m_loss16, arr_c_loss16, arr_s_loss16, arr_i_loss16, arr_sfm_loss16,  \
        precision_iCoseg16, precision_MSRC716, precision_int30016, precision_PAS16, jaccard_iCoseg16,\
        jaccard_MSRC716, jaccard_int30016, jaccard_PAS16 \
        = get_details1("datas/train16mse.log")

# precision 值和 epoch 值的可视化图像（已使用）
# image_visualization.precision_epoch_fig(precision_iCoseg2, precision_MSRC72, precision_int3002, precision_PAS2)

# Jaccard 值和 epoch 值的可视化图像
#image_visualization.jaccard_epoch_fig(jaccard_iCoseg, jaccard_MSRC7, jaccard_int300, jaccard_PAS)


# all_loss 值和 epoch 值的可视化图像（已使用）
#image_visualization.all_loss_epoch_fig(arr_all_loss8)

# 4个 loss 值和 epoch 值的可视化图像 效果差强人意（已使用）
# m_loss:WBCE loss, c_loss:classification loss, s_loss: 未知， i_loss: IOU loss
#image_visualization.four_loss_epoch_fig(arr_m_loss8, arr_c_loss8, arr_s_loss8, arr_i_loss8)

# batch size 和 precision 的可视化图像（已使用）
# image_visualization.bs_precision_fig(precision_MSRC72, precision_MSRC74, precision_MSRC78, precision_MSRC716)

# batch size 和 Jaccard 的可视化图像
#image_visualization.bs_jaccard_fig(jaccard_iCoseg2, jaccard_iCoseg4, jaccard_iCoseg8, jaccard_iCoseg16)

# 新加的 loss 和 epoch 的可视化图像（已使用）
#image_visualization.single_loss_epoch_fig(arr_sfm_loss8)

# 新加的 loss 和 batch size 的可视化图像
#image_visualization.sfm_loss_bs_fig(arr_sfm_loss2, arr_sfm_loss4, arr_sfm_loss8, arr_sfm_loss16)


# 获得最大的precision 值 和 Jaccard 值
print(get_max_evaluation_index(precision_iCoseg2, precision_MSRC72, precision_int3002, precision_PAS2))
print(get_max_evaluation_index(jaccard_iCoseg2, jaccard_MSRC72, jaccard_int3002, jaccard_PAS2))


# 散点图 数据集在不同方法上的 precision 值
# precision_original = [95.40, 93.30, 95.80, 97.60]
# precision_SFM = [95.34, 94.23, 95.68, 97.54]
# precision_MSE = [95.27, 93.91, 95.97, 97.54]

# image_visualization.sandiantu(precision_original,precision_SFM, precision_MSE)

