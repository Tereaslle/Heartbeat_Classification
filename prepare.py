import pandas as pd

if __name__ == '__main__':
    # 加载原始数据没有标题行需要自己设置
    df_column_name = ['id', 'heartbeat_signals', 'label']
    # 加载原始数据
    df_train = pd.read_csv('./data/train.csv')
    # df_testA = pd.read_csv('./data/testA.csv', header=None, names=['id', 'heartbeat_signals'])
    column_name = ['id']
    # 生成0~204心跳信号字段名
    for i in range(205):
        column_name.append(i)
    column_name.append('label')
    # 生成空dataframe含字段名
    data = pd.DataFrame(columns=column_name)
    # 读取1000行源数据，可以自己调整，刚开始可以少一点，利于Debug
    for index in range(500):
        series = df_train.iloc[index]   # 按列号获取数据，得到的是serious类型的变量
        series = series.values      # 取出List类型的值
        id = series[0]  # 第0列是id
        heartbeat_signal = series[1].split(',')     # 第1列是含有205个心跳信号的heartbeat_signal，类型是字符串，因此需要拆成单个变量，使用切片函数split
        heartbeat_signal = map(eval, heartbeat_signal)  # eval是一个函数根据字符串来计算结果，就是转为数字，map（eval，list）是将list内的所有元素作eval操作
        label = series[2]   # 第0列是label 标签
        # 重新拼接新的数据行
        t = []
        t.append(id)
        t.extend(heartbeat_signal)
        t.append(label)
        print(t)
        data.loc[index] = t
    '''
    path_or_buf : 文件路径，如果没有指定则将会直接返回字符串的 json
    sep : 输出文件的字段分隔符，默认为 “,”
    na_rep : 用于替换空数据的字符串，默认为''
    float_format : 设置浮点数的格式（几位小数点）
    columns : 要写的列
    header : 是否保存列名，默认为 True ，保存
    index : 是否保存索引，默认为 True ，保存
    index_label : 索引的列标签名
    '''
    # 修改dataframe的变量类型
    data = data.astype({'id': int,
                        'label': int})
    # print(type((data.loc[0])[0]))
    # 输出csv文件
    data.to_csv(path_or_buf='./data/prepared.csv', sep=',',header=False,index=False)