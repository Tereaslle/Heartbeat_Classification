from bpnn import BPNN

experience_metric = ["training accuracy", "test accuracy", "inference speed", "FPR"]

# 激活函数消融实验
def activation_ablation_study(times:int=10) -> None:
    '''
    :param times: 重复训练times次
    :return:
    '''
    # 对比5种激活函数的使用
    activation = [['relu', 'relu', 'relu', 'relu'],
                  ['leakyrelu', 'leakyrelu', 'leakyrelu', 'leakyrelu'],
                  ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'],
                  ['tanh', 'tanh', 'tanh', 'tanh'],
                  ['sigmoid', 'relu', 'tanh', 'tanh']]
    bpnn = BPNN()     # 实例化对象
    bpnn.readdata()  # 读取数据
    contrast_list = []  # 存放每个模型的评价结果，用于对比
    for a in activation:
        bpnn.activation = a  # 设置激活函数
        if 'leakyrelu' not in bpnn.activation:
            bpnn.init_model(bpnn.x_train)     # 根据激活函数初始化模型
        else:
            bpnn.init_model_with_leakyrelu(bpnn.x_train)      # leakyrelu需要单独初始化，因为它是高级激活函数
        metric = [0.0] * 4  # 评价指标
        for _ in range(times):
            bpnn.training()
            # 每训练一次，记录评价指标
            metric[0] += bpnn.trainHistory.history['accuracy'][-1]
            metric[1] += bpnn.score[1]
            metric[2] += bpnn.speed
            metric[3] += bpnn.fpr
        # 重复实验times次，对参数取平均后存入数组
        contrast_list.append(list(map(lambda x: x / times, metric)))
    # 输出表头
    print(f"{'activation':42}", end='')
    for i in experience_metric:
        print(f"{i:<22}", end='')
    print(' ')
    # 输出结果
    for i in range(len(contrast_list)):
        for a in activation[i]:
            print(f"{a:<10}", end='')
        print(': ', end='')
        for metric in contrast_list[i]:
            # <20表示左对齐20格宽度
            print(f"{metric:<22}", end='')
        print(' ')
    return

# 隐藏元个数消融函数
def unit_ablation_study(times: int=10) -> None:
    '''
    :param times: 重复训练times次
    :return:
    '''
    input_size = 205
    unit = [input_size*1, input_size*2, input_size*5, input_size*10]    # 对比第一层隐藏层的神经元个数（为输入元的整数倍）
    bpnn = BPNN()     # 实例化对象
    bpnn.readdata()  # 读取数据
    contrast_list = []  # 存放每个模型的评价结果，用于对比
    for u in unit:
        bpnn.units = [u, u // 2, u // 4, u // 8]   # 每层隐藏元个数是上一层的一半
        bpnn.init_model(bpnn.x_train)
        metric = [0.0] * 4      # 4类评价指标
        for _ in range(times):
            bpnn.training()
            # 每训练一次，记录评价指标
            metric[0] += bpnn.trainHistory.history['accuracy'][-1]
            metric[1] += bpnn.score[1]
            metric[2] += bpnn.speed
            metric[3] += bpnn.fpr
        # 重复实验times次，对参数取平均后存入数组
        contrast_list.append(list(map(lambda x: x / times, metric)))
    # 输出表头
    print(f"{'units':7}", end='')
    for i in experience_metric:
        print(f"{i:<22}", end='')
    print(' ')
    # 输出结果
    for i in range(len(unit)):
        print(f"{unit[i]:<5d}: ", end='')
        for metric in contrast_list[i]:
            # <20表示左对齐20格宽度
            print(f"{metric:<22}", end='')
        print(' ')
    return

if __name__ == '__main__':
    # 消融实验测试,注意修改重复实验次数times
    activation_ablation_study(times=1)
    unit_ablation_study(times=1)