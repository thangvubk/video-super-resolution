clear; close all;

%% Configurationi
% NOTE: you can modify this part
read_path = 'Test/Set5';
scale = 3;
use_upscale_interpolation = false;

%% Create save path for high resolution and low resolution images based on config
% NOTE: you should NOT modify the following parts

scale_dir = strcat(int2str(scale), 'x');

if use_upscale_interpolation
    interpolation_dir = 'interpolation';
else
    interpolation_dir = 'noninterpolation';
end

% example: hr_save_path = 'data/interpolation/Test/Set14/3x/high_res'
hr_save_path = fullfile('preprocessed_data', interpolation_dir, read_path, scale_dir, 'high_res');
lr_save_path = fullfile('preprocessed_data', interpolation_dir, read_path, scale_dir, 'low_res');

safe_mkdir(hr_save_path);
safe_mkdir(lr_save_path);

filepaths = dir(fullfile(read_path,'*.bmp'));
for i = 1 : length(filepaths)
    image = imread(fullfile(read_path,filepaths(i).name));
    if size(image, 3) == 3
        image_ycbcr = rgb2ycbcr(image);
        image_y = image_ycbcr(:, :, 1);
    end
    hr_im = im2double(image_y);
    hr_im = modcrop(hr_im, scale);
    [hei, wid] = size(hr_im);
    lr_im = imresize(hr_im,1/scale,'bicubic');
    
    if use_upscale_interpolation
        lr_im = imresize(lr_im ,[hei, wid],'bicubic');
    end
    
    imwrite(lr_im, fullfile(lr_save_path, filepaths(i).name))
    imwrite(hr_im, fullfile(hr_save_path, filepaths(i).name))
end

%% Utility function (supported in matlab 2016Rb or newer :D)
% NOTE: if your matlab version is lower than 2016Rb please copy modcrop
% 	and safe_mkdir to other .m file

function img = modcrop(img, scale)
% The img size should be divided by scale, to align interpolation
    sz = size(img);
    sz = sz - mod(sz, scale);
    img = img(1:sz(1), 1:sz(2));
end

function safe_mkdir(path)
% if the directory is exist, clean it
% else create it

    if exist(path, 'dir') == 7 % dir exists
        filepaths = dir(fullfile(path,'*.bmp'));
        for i = 1 : length(filepaths)
            file = fullfile(path,filepaths(i).name);
            delete(file);
        end
    else
        mkdir(path)
    end
end

