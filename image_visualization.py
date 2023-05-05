#
import plotly.graph_objs as go
import numpy as np

def precision_epoch_fig(precision_iCoseg, precision_MSRC7, precision_int300, precision_PAS):
    # 4个数据集上的 precision 和 epoch 的可视化图像 (固定batch_size，最好图中表明)
    # epoch的值
    #epoch = np.arange(1000, 100001, 1000)
    epoch = np.arange(1, 101, 1)

    # 创建每个数据集的trace
    trace1 = go.Scatter(x=epoch, y=precision_iCoseg, mode='lines', name='iCoseg8', line=dict(color='rgb(0, 0, 255)'))
    trace2 = go.Scatter(x=epoch, y=precision_MSRC7, mode='lines', name='MSRC7', line=dict(color='rgb(255, 0, 0)'))
    trace3 = go.Scatter(x=epoch, y=precision_int300, mode='lines', name='Int_300', line=dict(color='rgb(0, 255, 0)'))
    trace4 = go.Scatter(x=epoch, y=precision_PAS, mode='lines', name='PAS_VOC', line=dict(color='rgb(255, 165, 0)'))

    # 创建布局
    layout = go.Layout(title=dict(text="Precision vs. Epoch",font=dict(size=24), x=0.5),
                       xaxis=dict(title='Epoch/K', title_font=dict(size=18),
                                  tickfont=dict(size=16), range=[1, 100], dtick=10),
                       yaxis=dict(title='Precision',type='log', title_font=dict(size=18),
                                  tickfont=dict(size=16)),
                       legend=dict(font=dict(size=16)))

    # 创建Figure对象
    fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout)
    fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout)
    #fig.update_layout(height=600, width=800, margin=dict(l=50, r=50, t=80, b=50), showlegend=True)

    # 显示图像
    fig.show()

    return

def jaccard_epoch_fig(jaccard_iCoseg, jaccard_MSRC7, jaccard_int300, jaccard_PAS):
    # 4个数据集上的jaccard index 和 epoch 的可视化图像（固定batch_size，最好图中表明）
    epoch = np.arange(1, 101, 1)

    # 创建每个数据集的trace
    trace1 = go.Scatter(x=epoch, y=jaccard_iCoseg, mode='lines', name='iCoseg8', line=dict(color='rgb(0, 0, 255)'))
    trace2 = go.Scatter(x=epoch, y=jaccard_MSRC7, mode='lines', name='MSRC7', line=dict(color='rgb(255, 0, 0)'))
    trace3 = go.Scatter(x=epoch, y=jaccard_int300, mode='lines', name='Int_300', line=dict(color='rgb(0, 255, 0)'))
    trace4 = go.Scatter(x=epoch, y=jaccard_PAS, mode='lines', name='PAS_VOC', line=dict(color='rgb(255, 165, 0)'))

    # 创建布局
    layout = go.Layout(title=dict(text="Jaccard vs. Epoch(bs=4)", font=dict(size=24), x=0.5),
                       xaxis=dict(title='Epoch/K', title_font=dict(size=18),
                                  tickfont=dict(size=16), range=[1, 100], dtick=10),
                       yaxis=dict(title='Jaccard', type='log', title_font=dict(size=18),
                                  tickfont=dict(size=16)),
                       legend=dict(font=dict(size=16)))

    # 创建Figure对象
    fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout)

    # 显示图像
    fig.show()

    return

def all_loss_epoch_fig(arr_all_loss):
    # epoch的值
    epoch = np.arange(1, 1001, 1)

    # 创建每个数据集的trace
    trace1 = go.Scatter(x=epoch, y=arr_all_loss, mode='lines', name='all_loss', line=dict(color='rgb(0, 0, 255)'))

    # 创建布局
    layout = go.Layout(title=dict(text="change curve of all_loss(bs=8)", font=dict(size=24), x=0.5),
                       xaxis=dict(title='Epoch/H', title_font=dict(size=18),
                                  tickfont=dict(size=16)),
                       yaxis=dict(title='Loss', type='log', title_font=dict(size=18),
                                  tickfont=dict(size=16)),
                       legend=dict(font=dict(size=16)))

    # 创建Figure对象
    fig = go.Figure(data=[trace1], layout=layout)

    # 显示图像
    fig.show()

    return

def four_loss_epoch_fig(arr_m_loss, arr_c_loss, arr_s_loss, arr_i_loss):
    # 4个 loss 和 epoch 的可视化图像
    # epoch的值
    epoch = np.arange(1, 1001, 1)

    # 创建每个数据集的trace
    trace1 = go.Scatter(x=epoch, y=arr_m_loss, mode='lines', name='WBCE Loss', line=dict(color='rgb(0, 0, 255)'))
    trace2 = go.Scatter(x=epoch, y=arr_c_loss, mode='lines', name='cls Loss', line=dict(color='rgb(255, 0, 0)'))
    trace3 = go.Scatter(x=epoch, y=arr_s_loss, mode='lines', name='s Loss', line=dict(color='rgb(0, 255, 0)'))
    trace4 = go.Scatter(x=epoch, y=arr_i_loss, mode='lines', name='IOU Loss', line=dict(color='rgb(255, 165, 0)'))

    # 创建布局
    layout = go.Layout(title=dict(text="change curve of losses(bs=8)", font=dict(size=24), x=0.5),
                       xaxis=dict(title='Epoch/H', title_font=dict(size=18),
                                  tickfont=dict(size=16)),
                       yaxis=dict(title='Loss',type='log', title_font=dict(size=18),
                                  tickfont=dict(size=12), tickformat='.4f'),
                       legend=dict(font=dict(size=16)))

    # 创建Figure对象
    fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout)

    # 显示图像
    fig.show()

    return


def bs_precision_fig(arr_precision_2, arr_precision_4, arr_precision_8, arr_precision_16):

    # batch_size 和 precision 的可视化图像， 选定一个数据集
    epoch = np.arange(1, 101, 1)

    # 创建每个数据集的trace
    trace1 = go.Scatter(x=epoch, y=arr_precision_2, mode='lines', name='batch size = 2', line=dict(color='rgb(0, 0, 255)'))
    trace2 = go.Scatter(x=epoch, y=arr_precision_4, mode='lines', name='batch_size = 4', line=dict(color='rgb(255, 0, 0)'))
    trace3 = go.Scatter(x=epoch, y=arr_precision_8, mode='lines', name='batch_size = 8', line=dict(color='rgb(0, 255, 0)'))
    trace4 = go.Scatter(x=epoch, y=arr_precision_16, mode='lines', name='batch_size = 16', line=dict(color='rgb(255, 165, 0)'))

    # 创建布局
    layout = go.Layout(title=dict(text="Precision of different batch sizes", font=dict(size=24), x=0.5),
                       xaxis=dict(title='Epoch/K', title_font=dict(size=18),
                                  tickfont=dict(size=16), range=[1, 100], dtick=10),
                       yaxis=dict(title='Precision', type='log', title_font=dict(size=18),
                                  tickfont=dict(size=16)),
                       legend=dict(font=dict(size=16)))

    # 创建Figure对象
    fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout)

    # 显示图像
    fig.show()

    return

def bs_jaccard_fig(arr_jaccard_2, arr_jaccard_4, arr_jaccard_8, arr_jaccard_16):
    #batch_size 和 jaccard index 的可视化图像, 选定一个数据集

    epoch = np.arange(1, 101, 1)

    # 创建每个数据集的trace
    trace1 = go.Scatter(x=epoch, y=arr_jaccard_2, mode='lines', name='batch size = 2', line=dict(color='rgb(0, 0, 255)'))
    trace2 = go.Scatter(x=epoch, y=arr_jaccard_4, mode='lines', name='batch_size = 4', line=dict(color='rgb(255, 0, 0)'))
    trace3 = go.Scatter(x=epoch, y=arr_jaccard_8, mode='lines', name='batch_size = 8', line=dict(color='rgb(0, 255, 0)'))
    trace4 = go.Scatter(x=epoch, y=arr_jaccard_16, mode='lines', name='batch_size = 16', line=dict(color='rgb(255, 165, 0)'))

    # 创建布局
    layout = go.Layout(title=dict(text="Jaccard Index of different batch size", font=dict(size=24), x=0.5),
                       xaxis=dict(title='Epoch/K', title_font=dict(size=18),
                                  tickfont=dict(size=16), range=[1, 100], dtick=10),
                       yaxis=dict(title='Jaccard', type='log', title_font=dict(size=18),
                                  tickfont=dict(size=16)),
                       legend=dict(font=dict(size=16)))

    # 创建Figure对象
    fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout)

    # 显示图像
    fig.show()

    return

#--------------------------------------------------------------------------------------------------------------

def single_loss_epoch_fig(loss_arr):
    #探究单个 loss 和 epoch 的可视化图像（适用于：sfm loss 和 mse loss）,观察函数图像的收敛，从而便于进行下一步的操作
    # epoch的值
    epoch = np.arange(1, 1001, 1)

    # 创建每个数据集的trace
    trace1 = go.Scatter(x=epoch, y=loss_arr, mode='lines', name='loss_name', line=dict(color='rgb(0, 0, 255)'))

    # 创建布局
    layout = go.Layout(title=dict(text="change curve of mse loss(bs=8)", font=dict(size=24), x=0.5),
                       xaxis=dict(title='Epoch/H', title_font=dict(size=18),
                                  tickfont=dict(size=16)),
                       yaxis=dict(title='Loss', type='log', title_font=dict(size=18),
                                  tickfont=dict(size=16)),
                       legend=dict(font=dict(size=16)))

    # 创建Figure对象
    fig = go.Figure(data=[trace1], layout=layout)

    # 显示图像
    fig.show()

    return

def sfm_precision_fig():
    # 探究 在不同的 bs 下， sfm loss 对于 precision 值的影响，考虑后认为使用表格进行阐述会更加地直接。
    # 使用 bs_precision_fig() 即可。
    return


def sfm_loss_bs_fig(arr_loss_2, arr_loss_4, arr_loss_8, arr_loss_16):
    # 探究不同的 batch size 大小对于 sfm loss 值的影响下，发现很少发生收敛，探究：不太合适（表格阐述可能会更加合适）
    # 考虑更换 loss 函数
    epoch = np.arange(1, 1001, 1)

    # 创建每个数据集的trace
    trace1 = go.Scatter(x=epoch, y=arr_loss_2, mode='lines', name='batch size = 2', line=dict(color='rgb(0, 0, 255)'))
    trace2 = go.Scatter(x=epoch, y=arr_loss_4, mode='lines', name='batch size = 4', line=dict(color='rgb(255, 0, 0)'))
    trace3 = go.Scatter(x=epoch, y=arr_loss_8, mode='lines', name='batch size = 8', line=dict(color='rgb(0, 255, 0)'))
    trace4 = go.Scatter(x=epoch, y=arr_loss_16, mode='lines', name='batch size = 16', line=dict(color='rgb(255, 165, 0)'))

    # 创建布局
    layout = go.Layout(title=dict(text="sfm loss with different batch sizes", font=dict(size=24), x=0.5),
                       xaxis=dict(title='Epoch/H', title_font=dict(size=18),
                                  tickfont=dict(size=16), range=[1, 800]),
                       yaxis=dict(title='Loss', type='log', title_font=dict(size=18),
                                  tickfont=dict(size=12), tickformat='.4f'),
                       legend=dict(font=dict(size=16)))

    # 创建Figure对象
    fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout)

    # 显示图像
    fig.show()

    return

def loss_function_precision_fig(arr_precision_sfm, arr_precision_mse):
    # 探究添加不同的 loss 函数（sfm 和 mse) 对于最后结果 Precision 的影响（可基于不同的 batch size）
    epoch = np.arange(1, 101, 1)

    # 创建每个数据集的trace
    trace1 = go.Scatter(x=epoch, y=arr_precision_sfm, mode='lines', name='+sfm',
                        line=dict(color='rgb(0, 0, 255)'))
    trace2 = go.Scatter(x=epoch, y=arr_precision_mse, mode='lines', name='+mse',
                        line=dict(color='rgb(255, 0, 0)'))
    # trace3 = go.Scatter(x=epoch, y=arr_precision_8, mode='lines', name='batch_size = 8',
    #                     line=dict(color='rgb(0, 255, 0)'))
    # trace4 = go.Scatter(x=epoch, y=arr_precision_16, mode='lines', name='batch_size = 16',
    #                     line=dict(color='rgb(255, 165, 0)'))

    # 创建布局
    layout = go.Layout(title=dict(text="Precision with different loss functions", font=dict(size=24), x=0.5),
                       xaxis=dict(title='Epoch/K', title_font=dict(size=18),
                                  tickfont=dict(size=16), range=[1, 100], dtick=10),
                       yaxis=dict(title='Precision', type='log', title_font=dict(size=18),
                                  tickfont=dict(size=16)),
                       legend=dict(font=dict(size=16)))

    # 创建Figure对象
    fig = go.Figure(data=[trace1, trace2], layout=layout)

    # 显示图像
    fig.show()

    return

def loss_function_jaccard_fig(arr_jaccard_sfm, arr_jaccard_mse):
    # 探究添加不同的 loss 函数（sfm 和 mse）对于最后结果 Jaccard 的影响（可基于不同的 batch size）
    epoch = np.arange(1, 101, 1)

    # 创建每个数据集的trace
    trace1 = go.Scatter(x=epoch, y=arr_jaccard_sfm, mode='lines', name='+sfm',
                        line=dict(color='rgb(0, 0, 255)'))
    trace2 = go.Scatter(x=epoch, y=arr_jaccard_mse, mode='lines', name='+mse',
                        line=dict(color='rgb(255, 0, 0)'))
    # trace3 = go.Scatter(x=epoch, y=arr_precision_8, mode='lines', name='batch_size = 8',
    #                     line=dict(color='rgb(0, 255, 0)'))
    # trace4 = go.Scatter(x=epoch, y=arr_precision_16, mode='lines', name='batch_size = 16',
    #                     line=dict(color='rgb(255, 165, 0)'))

    # 创建布局
    layout = go.Layout(title=dict(text="Jaccard Index with different loss functions", font=dict(size=24), x=0.5),
                       xaxis=dict(title='Epoch/K', title_font=dict(size=18),
                                  tickfont=dict(size=16), range=[1, 100], dtick=10),
                       yaxis=dict(title='Jaccard', type='log', title_font=dict(size=18),
                                  tickfont=dict(size=16)),
                       legend=dict(font=dict(size=16)))

    # 创建Figure对象
    fig = go.Figure(data=[trace1, trace2], layout=layout)

    # 显示图像
    fig.show()

    return


def sandiantu(arr_precision_ori, arr_precision_sfm, arr_precision_mse):
    # 创建每个数据集的trace
    batch_size = [  "PAS_VOC", "Int_300","MSRC7","iCoseg" ]
    trace1 = go.Scatter(x=batch_size, y=arr_precision_ori, mode='lines+markers', name='UFO', line=dict(color='rgb(0, 0, 255)'))
    trace2 = go.Scatter(x=batch_size, y=arr_precision_sfm, mode='lines+markers', name='UFO + SFM', line=dict(color='rgb(255, 0, 0)'))
    trace3 = go.Scatter(x=batch_size, y=arr_precision_mse, mode='lines+markers', name='UFO + MSE', line=dict(color='rgb(0, 255, 0)'))
    #trace4 = go.Scatter(x=epoch, y=arr_loss_16, mode='lines', name='batch size = 16',
    #                    line=dict(color='rgb(255, 165, 0)'))

    # 创建布局
    layout = go.Layout(title=dict(text="precisions with different datasets and models", font=dict(size=24), x=0.5),
                       xaxis=dict(title='datasets', title_font=dict(size=18),
                                  tickfont=dict(size=16), range=[1, 800]),
                       yaxis=dict(title='Precision',  title_font=dict(size=18),
                                  tickfont=dict(size=12), tickformat='.4f'),
                       legend=dict(font=dict(size=16)))

    # 创建Figure对象
    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)

    # 显示图像
    fig.show()

    return
