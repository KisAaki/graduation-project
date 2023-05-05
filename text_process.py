import re

# # 读取文本文件
# with open('train4sfm.log', 'r') as f:
#     text = f.read()
#
#
# # 用正则表达式匹配每一行中的数据 未添加额外的损失函数
# #pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} lr: (\d+\.\d+e?-?\d+), epoch: \[(\d+)/\d+\], all_loss: \[(\d+\.\d+)\], m_loss: \[(\d+\.\d+)\], c_loss: \[(\d+\.\d+)\], s_loss: \[(\d+\.\d+)\], i_loss: \[(\d+\.\d+)\]"
#
# #添加 sfm_loss
# pattern = r'lr: (\d+\.\d+e?-?\d+), epoch: \[(\d+)/\d+\], all_loss: \[(\d+\.\d+)\], m_loss: \[(\d+\.\d+)\], c_loss: \[(\d+\.\d+)\], s_loss: \[(\d+\.\d+)\], i_loss: \[(\d+\.\d+)\], sfm_loss:\[(\d+\.\d+)\]'
# results = re.findall(pattern, text)
#
# # 打印结果
# for lr, epoch, all_loss, m_loss, c_loss, s_loss, i_loss, sfm_loss in results:
#     print(f"lr: {lr}, epoch: {epoch}, all_loss: {all_loss}, m_loss: {m_loss}, c_loss: {c_loss}, s_loss: {s_loss}, i_loss: {i_loss}, sfm_loss: {sfm_loss}")
#
#
# # 用正则表达式匹配每一行中的数据
# p_pattern = r"p: \[(\d+\.\d+)\]"
# j_pattern = r"j: \[(\d+\.\d+)\]"
#
# p_values = re.findall(p_pattern, text)
# j_values = re.findall(j_pattern, text)
#
# print("P values:", p_values)
# print("J values:", j_values)


def get_details(file_name):


    with open(file_name, 'r') as f:
        text = f.read()

    # 用正则表达式匹配每一行中的数据 未添加额外的损失函数
    pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} lr: (\d+\.\d+e?-?\d+), epoch: \[(\d+)/\d+\], all_loss: \[(\d+\.\d+)\], m_loss: \[(\d+\.\d+)\], c_loss: \[(\d+\.\d+)\], s_loss: \[(\d+\.\d+)\], i_loss: \[(\d+\.\d+)\]"

    # 添加 sfm_loss 正则
    # pattern = r'lr: (\d+\.\d+e?-?\d+), epoch: \[(\d+)/\d+\], all_loss: \[(\d+\.\d+)\], m_loss: \[(\d+\.\d+)\], c_loss: \[(\d+\.\d+)\], s_loss: \[(\d+\.\d+)\], i_loss: \[(\d+\.\d+)\], sfm_loss:\[(\d+\.\d+)\]'
    results = re.findall(pattern, text)

    arr_lr = []
    arr_epoch = []
    arr_all_loss = []
    arr_m_loss = []
    arr_c_loss = []
    arr_s_loss = []
    arr_i_loss = []

    # 打印结果
    for lr, epoch, all_loss, m_loss, c_loss, s_loss, i_loss in results:
    #for lr, epoch, all_loss, m_loss, c_loss, s_loss, i_loss, sfm_loss in results:
        arr_lr.append(lr)
        arr_epoch.append(epoch)
        arr_all_loss.append(all_loss)
        arr_m_loss.append(m_loss)
        arr_c_loss.append(c_loss)
        arr_s_loss.append(s_loss)
        arr_i_loss.append(i_loss)
        #arr_sfm_loss.append(sfm_loss)

        # print(
        #     f"lr: {lr}, epoch: {epoch}, all_loss: {all_loss}, m_loss: {m_loss}, c_loss: {c_loss}, s_loss: {s_loss}, i_loss: {i_loss}, sfm_loss: {sfm_loss}")

    # 用正则表达式匹配每一行中的数据
    p_pattern = r"p: \[(\d+\.\d+)\]"
    j_pattern = r"j: \[(\d+\.\d+)\]"

    p_values = re.findall(p_pattern, text)
    j_values = re.findall(j_pattern, text)

    precision_iCoseg = []
    precision_MSRC7 = []
    precision_int300 = []
    precision_PAS = []

    jaccard_iCoseg = []
    jaccard_MSRC7 = []
    jaccard_int300 = []
    jaccard_PAS = []

    for i in range(0, 100, 1):
        precision_iCoseg.append(p_values[i * 4 + 0])
        precision_MSRC7.append(p_values[i * 4 + 1])
        precision_int300.append(p_values[i * 4 + 2])
        precision_PAS.append(p_values[i * 4 + 3])

        jaccard_iCoseg.append(j_values[i * 4 + 0])
        jaccard_MSRC7.append(j_values[i * 4 + 1])
        jaccard_int300.append(j_values[i * 4 + 2])
        jaccard_PAS.append(j_values[i * 4 + 3])


    #print("P values:", p_values)
    #print("J values:", j_values)

    #新加的loss，没有则取值为[]，空

    return arr_lr, arr_epoch, arr_all_loss, arr_m_loss, arr_c_loss, arr_s_loss, arr_i_loss, \
        precision_iCoseg, precision_MSRC7, precision_int300, precision_PAS, jaccard_iCoseg, jaccard_MSRC7, jaccard_int300, jaccard_PAS

    #return arr_lr, arr_epoch, arr_all_loss, arr_m_loss, arr_c_loss, arr_s_loss, arr_i_loss, arr_sfm_loss, \
    #    precision_iCoseg, precision_MSRC7, precision_int300, precision_PAS, jaccard_iCoseg, jaccard_MSRC7, jaccard_int300, jaccard_PAS


def get_details1(file_name):


    with open(file_name, 'r') as f:
        text = f.read()

    # 用正则表达式匹配每一行中的数据 未添加额外的损失函数
    # pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} lr: (\d+\.\d+e?-?\d+), epoch: \[(\d+)/\d+\], all_loss: \[(\d+\.\d+)\], m_loss: \[(\d+\.\d+)\], c_loss: \[(\d+\.\d+)\], s_loss: \[(\d+\.\d+)\], i_loss: \[(\d+\.\d+)\]"

    # 添加 sfm_loss 正则
    # pattern = r'lr: (\d+\.\d+e?-?\d+), epoch: \[(\d+)/\d+\], all_loss: \[(\d+\.\d+)\], m_loss: \[(\d+\.\d+)\], c_loss: \[(\d+\.\d+)\], s_loss: \[(\d+\.\d+)\], i_loss: \[(\d+\.\d+)\], sfm_loss:\[(\d+\.\d+)\]'

    #添加 mse_loss 正则
    pattern = r'lr: (\d+\.\d+e?-?\d+), epoch: \[(\d+)/\d+\], all_loss: \[(\d+\.\d+)\], m_loss: \[(\d+\.\d+)\], c_loss: \[(\d+\.\d+)\], s_loss: \[(\d+\.\d+)\], i_loss: \[(\d+\.\d+)\], mse_loss:\[(\d+\.\d+)\]'
    results = re.findall(pattern, text)

    arr_lr = []
    arr_epoch = []
    arr_all_loss = []
    arr_m_loss = []
    arr_c_loss = []
    arr_s_loss = []
    arr_i_loss = []
    arr_sfm_loss = []
    print(results)
    # 打印结果
    #for lr, epoch, all_loss, m_loss, c_loss, s_loss, i_loss in results:
    for lr, epoch, all_loss, m_loss, c_loss, s_loss, i_loss, mse_loss in results:
        arr_lr.append(lr)
        arr_epoch.append(epoch)
        arr_all_loss.append(all_loss)
        arr_m_loss.append(m_loss)
        arr_c_loss.append(c_loss)
        arr_s_loss.append(s_loss)
        arr_i_loss.append(i_loss)
        arr_sfm_loss.append(mse_loss)

        # print(
        #     f"lr: {lr}, epoch: {epoch}, all_loss: {all_loss}, m_loss: {m_loss}, c_loss: {c_loss}, s_loss: {s_loss}, i_loss: {i_loss}, sfm_loss: {sfm_loss}")

    # 用正则表达式匹配每一行中的数据
    p_pattern = r"p: \[(\d+\.\d+)\]"
    j_pattern = r"j: \[(\d+\.\d+)\]"

    p_values = re.findall(p_pattern, text)
    j_values = re.findall(j_pattern, text)

    precision_iCoseg = []
    precision_MSRC7 = []
    precision_int300 = []
    precision_PAS = []

    jaccard_iCoseg = []
    jaccard_MSRC7 = []
    jaccard_int300 = []
    jaccard_PAS = []

    for i in range(0, 100, 1):
        precision_iCoseg.append(p_values[i * 4 + 0])
        precision_MSRC7.append(p_values[i * 4 + 1])
        precision_int300.append(p_values[i * 4 + 2])
        precision_PAS.append(p_values[i * 4 + 3])

        jaccard_iCoseg.append(j_values[i * 4 + 0])
        jaccard_MSRC7.append(j_values[i * 4 + 1])
        jaccard_int300.append(j_values[i * 4 + 2])
        jaccard_PAS.append(j_values[i * 4 + 3])


    #print("P values:", p_values)
    #print("J values:", j_values)

    #新加的loss，没有则取值为[]，空

    #return arr_lr, arr_epoch, arr_all_loss, arr_m_loss, arr_c_loss, arr_s_loss, arr_i_loss, \
    #    precision_iCoseg, precision_MSRC7, precision_int300, precision_PAS, jaccard_iCoseg, jaccard_MSRC7, jaccard_int300, jaccard_PAS

    return arr_lr, arr_epoch, arr_all_loss, arr_m_loss, arr_c_loss, arr_s_loss, arr_i_loss, arr_sfm_loss, \
        precision_iCoseg, precision_MSRC7, precision_int300, precision_PAS, jaccard_iCoseg, jaccard_MSRC7, jaccard_int300, jaccard_PAS



def get_max_evaluation_index(index_iCoseg8, index_MSRC7, index_int300, index_PASVOC):
    max_index_iCoseg8 = max(index_iCoseg8)
    max_index_MSRC7 = max(index_MSRC7)
    max_index_int300 = max(index_int300)
    max_index_PASVOC = max(index_PASVOC)

    return max_index_iCoseg8, max_index_MSRC7, max_index_int300, max_index_PASVOC