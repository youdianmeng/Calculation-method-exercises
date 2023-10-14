def divided_difference(x, y):
    # x, y 分别为样本点坐标向量
    # 检查输入
    if len(x) != len(y):
        raise ValueError("The lengths of x and y must be equal")
    if len(x) == 0:
        raise ValueError("The inputs cannot be empty")
    # 构建列表
    table = [[xi, yi] for xi, yi in zip(x, y)]
    # 输入n阶，有n-1阶差商
    for k in range(1, len(x)):
        # 每一阶差商有个数与差商阶数和为n
        for i in range(len(x) - k):
            # 计算
            diff = (table[i + 1][k] - table[i][k]) / (table[i + 1][0] - table[i][0])
            # 确保差商在最上层保证循环
            table[i].append(diff)
    # 返回值
    return table
# 测试
x = [0.40, 0.55, 0.65, 0.80, 0.90]
y = [0.41075, 0.57815, 0.69675, 0.88811, 1.02652]
table = divided_difference(x, y)

for row in table:
   print(row)
