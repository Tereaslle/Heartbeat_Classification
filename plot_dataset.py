import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df_train = pd.read_csv('./data/train.csv')
    # 绘制每种类别的折线图
    ids = []
    for label, row in df_train.groupby('label').apply(lambda x: x.iloc[2, :]).iterrows():
        # 这里apply(lambda x: x.iloc[2,:])是对每个group的组选择下标为2的数据行
        ids.append(int(label))
        signals = list(map(float, row['heartbeat_signals'].split(',')))
        plt.plot(range(len(signals)), signals)

    # 设置字体为楷体 以下for循环查看有哪些字体可以用
    # for font in plt.font_manager.fontManager.ttflist:
    #     # 查看字体名以及对应的字体文件名
    #     print(font.name, '-', font.fname)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.title("四类标签心跳信号折线图")
    plt.xlabel("采样点编号")
    plt.ylabel("信号值")
    plt.legend(ids)
    plt.show()
