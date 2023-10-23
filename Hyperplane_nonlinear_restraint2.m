function [c, ceq] = Hyperplane_nonlinear_restraint2(alpha, beta_plus_minus, X_plus, X_minus, r_plus_minus)
    %% 超平面参数非线性约束函数
    % alpha: 超平面参数权重向量，包括alpha_plus and alpha_minus, alpha_plus在前
    % beta_plus_minus: 类别i和j的拉格朗日系数beta
    % X_plus: 正样本集合
    % X_minus: 负样本集合
    % r_plus_minus：正负样本求解出的超圆盘的r
    
    M_plus = size(X_plus, 1);
    alpha_plus = alpha(1:M_plus);
    alpha_minus = alpha(M_plus+1, end);
    M_minus = length(alpha_minus);
    beta_plus = beta_plus_minus(1:M_plus);
    beta_minus = beta_plus_minus(1:M_minus);
    r_plus = r_plus_minus(1);
    r_minus = r_plus_minus(2);
    
    c = zeros(2, 1);
    % 求解c(1)
    for i = 1 : M_plus
        c(1) = c(1) + sum(alpha_plus(i) * alpha_plus .* Gaussian_kernel_function(X_plus(i, :), X_plus, 0.25)) ...
            - sum(alpha_plus(i) * beta_plus .* Gaussian_kernel_function(X_plus(i, :), X_plus, 0.25)) ...
            + sum(beta_plus(i) * beta_plus .* Gaussian_kernel_function(X_plus(i, :), X_plus, 0.25)) ;
    end
    c(1) = c(1) - r_plus^2;

    % 求解c(2)
    for i = 1 : M_minus
        c(2) = c(2) + sum(alpha_minus(i) * alpha_minus .* Gaussian_kernel_function(X_minus(i, :), X_minus, 0.25)) ...
            - sum(alpha_minus(i) * beta_minus .* Gaussian_kernel_function(X_minus(i, :), X_minus, 0.25)) ...
            + sum(beta_minus(i) * beta_minus .* Gaussian_kernel_function(X_minus(i, :), X_minus, 0.25)) ;
    end
    c(2) = c(2) - r_minus^2;

    ceq = [];

end
