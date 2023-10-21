function True_or_false = RDS_restraint(X)
    %% 随机方向搜索的约束函数
    % X 为多维函数值某点坐标

    restraint1 = 9 - X(1)^2 - X(2)^2;
    restraint2 = X(1) + X(2) - 1;
    % fprintf('restraint1=%f; ', restraint1);
    % fprintf('restraint2=%f; \n', restraint2);
    if restraint1 >= 0 && restraint2 <= 0
        True_or_false = 1;
    else
        True_or_false = 0;
    end

end