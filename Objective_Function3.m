function result = Objective_Function3(alpha, X_plus, X_minus)
    %% 求解分类超平面参数权重向量的二次约束二次规划的目标函数,使用高斯核函数
    % alpha: 为该输入样本集合下的分类超平面参数
    % X_plus: X_plus为输入正样本矩阵，行为样本个数，列为样本特征
    % X_minus: X_minus为输入负样本矩阵，行为样本个数，列为样本特征
    
    M_plus = size(X_plus, 1);
    alpha_plus = alpha(1:M_plus);
    alpha_minus = alpha(M_plus+1, end);
    M_minus = length(alpha_minus);
    
    result = 0;
    % 第一、第三循环可以合并
    for i = 1 : M_plus
        result = result + sum(alpha_plus(i) * alpha_plus .* Gaussian_kernel_function(X_plus(i, :), X_plus, 0.25));  
    end
    for i = 1 : M_minus
        result = result + sum(alpha_minus(i) * alpha_minus .*  Gaussian_kernel_function(X_minus(i, :), X_minus, 0.25));
    end
    for i = 1 : M_plus
        result = result - 2 * sum(alpha_plus(i) * alpha_minus .* Gaussian_kernel_function(X_plus(i, :), X_minus, 0.25));  
    end

end