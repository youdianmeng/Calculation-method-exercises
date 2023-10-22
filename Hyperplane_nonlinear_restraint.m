function [c, ceq] = Hyperplane_nonlinear_restraint(alpha, X_plus, X_minus, s_and_r_plus_minus)
    %% 超平面参数非线性约束函数
    % alpha: 超平面参数权重向量，包括alpha_plus and alpha_minus, alpha_plus在前
    % X_plus: 正样本集合
    % X_minus: 负样本集合
    % s_and_r_plus_minus：正负样本求解出的超圆盘的s和r

    M_plus = size(X_plus, 1);
    alpha_plus = alpha(1:M_plus);
    alpha_minus = alpha(M_plus+1, end);
    s_plus = s_and_r_plus_minus(1, 1:end-1);
    s_minus = s_and_r_plus_minus(2, 1:end-1);
    r_plus = s_and_r_plus_minus(1, end);
    r_minus = s_and_r_plus_minus(2, end);

    c = zeros(2, 1);

    c(1) = (norm(sum(alpha_plus.*X_plus, 1) - s_plus))^2 - r_plus^2;
    c(2) = (norm(sum(alpha_minus.*X_minus, 1) - s_minus))^2 - r_minus^2;
    ceq = [];

end

