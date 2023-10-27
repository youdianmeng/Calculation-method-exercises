folder = 'F:\Dateset\SEU_dataset\gearbox\gearset'; % 你的文件夹路径
files = dir(fullfile(folder, '*.csv')); % 获取文件夹中所有以.csv结尾的文件

data = cell(1, length(files)); % 创建一个cell数组来存储每个文件的数据

for i = 1:length(files)
    filename = fullfile(folder, files(i).name); % 获取当前文件的完整路径
    data{i} = readtable(filename); % 假设文件是以.csv格式存储的，可以使用csvread函数读取数据
end
