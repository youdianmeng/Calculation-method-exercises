function [segmented_data, Class_] = make_dataset(data_src, num_samples, class_)

    % 获取目标文件夹数据
    find_file = fullfile(data_src, '/*.mat');
    files = dir(find_file);
    files = sort({files.name});
    data = load(fullfile(data_src, files{1})); % struct数据类型
    keysList = fieldnames(data);
    my_key = '';
    for i = 1:length(keysList)
        if contains(keysList{i}, '_DE_time')
            my_key = keysList{i};
            break;
        end
    end
    drive_end_data = data.(my_key);
    drive_end_data = drive_end_data(:);

    % 分割数据块
    num_segments = floor(length(drive_end_data)/num_samples);
    slices = drive_end_data(1:num_segments*num_samples);
    segmented_data = reshape(slices, [num_samples, num_segments])'; % matlab的reshape是按列依此重组的

    % 预防性操作
    files = files(2:end);
    for i = 1:length(files)
        data = load(fullfile(data_src, files{i}));
        keysList = fieldnames(data);
        for j = 1:length(keysList)
            if contains(keysList{j}, '_DE_time')
                my_key = keysList{j};
            end
        end
        drive_end_data = data.(my_key);
        drive_end_data = drive_end_data(:);
        num_segments = floor(length(drive_end_data)/num_samples);
        slices = drive_end_data(1:num_segments*num_samples);
        segmented_data = [segmented_data; reshape(slices, [num_samples, num_segments])'];
    end
    
    segmented_data = unique(segmented_data, 'rows'); % remove duplicates
    segmented_data = segmented_data(randperm(size(segmented_data, 1)), :); % shuffle the data
    Class_ = ones(size(segmented_data, 1), 1) * class_;

end
