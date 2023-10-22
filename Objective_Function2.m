function result = Objective_Function2(alpha, X_plus, X_minus)
    %% 求解分类超平面参数权重向量的二次约束二次规划的目标函数
    % alpha: 为该输入样本集合下的分类超平面参数
    % X_plus: X_plus为输入正样本矩阵，行为样本个数，列为样本特征
    % X_minus: X_minus为输入负样本矩阵，行为样本个数，列为样本特征
    
    M_plus = size(X_plus, 1);
    result = (norm(sum(alpha(1:M_plus).*X_plus, 1) - sum(alpha(M_plus+1, :).*X_minus))) ^ 2;

end