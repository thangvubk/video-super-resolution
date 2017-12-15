clear; close all;

%% Configurationi
% NOTE: you can modify this part
read_path = 'test';
scale = 3;
width = 960;
height = 540;
use_upscale_interpolation = true;


%% Create save path for high resolution and low resolution images based on config
% NOTE: you should NOT modify the following parts

scale_dir = strcat(int2str(scale), 'x');

if use_upscale_interpolation
    interpolation_dir = 'interpolation';
else
    interpolation_dir = 'noninterpolation';
end

% example: hr_save_path = 'data/interpolation/test/3x/high_res'
save_path = fullfile('preprocessed_data_video', read_path, interpolation_dir, scale_dir);

data = zeros(height, width, 5, 1);
label = zeros(height, width, 5, 1);

dirs = dir(read_path);

count = 0;

for i_dir = 1 : length(dirs)
    scene_dir = dirs(i_dir).name;
    if strcmp(scene_dir, '.') || strcmp(scene_dir, '..')
        continue
    end
    count = count + 1;
    
    filepaths = dir(fullfile(read_path, scene_dir, '*.bmp'));
    
    for i = 1 : length(filepaths)
        image = imread(fullfile(read_path, scene_dir, filepaths(i).name));
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
        
        data(:, :, i, count) = lr_im;
        label(:, :, i, count) = hr_im;
    end
end

%% writing to HDF5
chunksz = 2;
created_flag = false;
totalct = 0;

for batchno = 1:floor((count)/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
    batchlabs = label(:,:,:,last_read+1:last_read+chunksz);

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(fullfile(save_path, 'dataset.h5'), batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(fullfile(save_path, 'dataset.h5'));

%% Utility function (supported in matlab 2016Rb or newer :D)
% NOTE: if your matlab version is lower than 2016Rb please copy modcrop
% to other .m file

function img = modcrop(img, scale)
% The img size should be divided by scale, to align interpolation
    sz = size(img);
    sz = sz - mod(sz, scale);
    img = img(1:sz(1), 1:sz(2));
end


