clear; close all;

%% Configuration 
% NOTE: you can modify this part
read_path = 'Train';
scale = 3;
use_upscale_interpolation = false;
hr_size = 33;
stride = 4*scale;


%% Create save path for high resolution and low resolution images based on config
% NOTE: you should NOT modify the following parts

scale_dir = strcat(int2str(scale), 'x');
if use_upscale_interpolation
    interpolation_dir = 'interpolation';
else
    interpolation_dir = 'noninterpolation';
end

% example: hr_save_path = 'data/noninterpolation/Test/Set14/3x/high_res'
hr_save_path = fullfile('data', interpolation_dir, read_path, scale_dir, 'high_res');
lr_save_path = fullfile('data', interpolation_dir, read_path, scale_dir, 'low_res');

safe_mkdir(hr_save_path)
safe_mkdir(lr_save_path)

count = 0;
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
    
    for h = 1 : stride : hei - hr_size + 1
        for w = 1 : stride : wid - hr_size + 1
            
            hr_sub_im = hr_im(h:hr_size+h-1, w:hr_size+w-1);
                
            if use_upscale_interpolation
                lr_sub_im = lr_im(h:hr_size+h-1, w:hr_size+w-1);
            else
                lr_sub_im = lr_im(uint32((h-1)/scale + 1):uint32((hr_size+h-1)/scale), uint32((w-1)/scale+1):uint32((hr_size+w-1)/scale));
            end
            
            count = count + 1;
            
            imwrite(lr_sub_im, fullfile(lr_save_path, strcat(sprintf('%08d', count), '.bmp')))
            imwrite(hr_sub_im, fullfile(hr_save_path, strcat(sprintf('%08d', count), '.bmp')))
        end
    end
    
    
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
