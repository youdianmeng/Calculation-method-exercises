function result = Objective_Function1(beta, X)
    %% 返回超圆盘的对偶问题的目标函数，在某一拉格朗日系数下的值
    % X: X为输入样本矩阵，行为样本个数，列为样本特征
    % beta: 为该输入样本集合下的拉格朗日系数
    
    result = sum(beta .* sum(X.*X, 2)) - sum(sum((beta.*X)*(beta.*X)', 2));

end