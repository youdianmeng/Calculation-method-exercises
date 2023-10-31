% 指定文件夹路径
folder_path = 'F:\python_learning\Calculation-method-exercises\pretreatment_HD\SEU_ed';
% 获取文件夹中所有.txt文件的信息
files = dir(fullfile(folder_path, '*.csv'));

% 定义一个cell数组，用于存储所有文件的数据
data = cell(1, length(files));

% 遍历每个.txt文件
for i = 1:length(files)
    % 获取文件名（包括扩展名）
    filename = files(i).name;
    % 打印文件名
    data{1} = readtable(filename);
    

end
