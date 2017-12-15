clear; close all;

%% Configuration 
% NOTE: you can modify this part
train_set = 'train';
scale = 3;
use_upscale_interpolation = true;
hr_size = 48;
stride = 24;

%% Create save path for high resolution and low resolution images based on config
% NOTE: you should NOT modify the following parts
disp(sprintf('%10s: %s', 'Train set', train_set));
disp(sprintf('%10s: %d', 'Scale', scale));

scale_dir = strcat(int2str(scale), 'x');

% example: 
% read_path = '../data/train'
% save_path = '../preprocessed_data_video/train/3x/'
read_path = fullfile('../data', train_set) 
save_path = fullfile('../preprocessed_data', train_set, scale_dir);

if exist(save_path, 'dir') ~= 7
	mkdir(save_path)
end

% count variable to order the data
base_count = 0;
count = 0;

data = zeros(hr_size, hr_size, 5, 1);
label = zeros(hr_size, hr_size, 5, 1);

dirs = dir(read_path);
for i_dir = 1 : length(dirs)
    is_switch_dir = true;
    scene_dir = dirs(i_dir).name;
    if scene_dir(1) ~= 's' %valid folder begin with 's'
        continue
    end
    disp(sprintf('processing dir: %s', scene_dir));
    
    filepaths = dir(fullfile(read_path, scene_dir, '*.bmp'));
    
    for i = 1 : length(filepaths)
        % if switch dir add count to base_count
        if is_switch_dir
            base_count = base_count + count;
            is_switch_dir = false;
        end
        
        % reset count
        count = 0;
        
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

        for h = 1 : stride : hei - hr_size + 1
            for w = 1 : stride : wid - hr_size + 1

                hr_sub_im = hr_im(h:hr_size+h-1, w:hr_size+w-1);

                if use_upscale_interpolation
                    lr_sub_im = lr_im(h:hr_size+h-1, w:hr_size+w-1);
                else
                    lr_sub_im = lr_im(uint32((h-1)/scale + 1):uint32((hr_size+h-1)/scale), uint32((w-1)/scale+1):uint32((hr_size+w-1)/scale));
                end

                count = count + 1;

                data(:, :, i, base_count + count) = lr_sub_im;
                label(:, :, i, base_count + count) = hr_sub_im;
            end
        end


    end
end

order = randperm(base_count + count);
data = data(:, :, :, order);
label = label(:, :, :, order); 

%% writing to HDF5
chunksz = 32;
created_flag = false;
totalct = 0;

for batchno = 1:floor((base_count + count)/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
    batchlabs = label(:,:,:,last_read+1:last_read+chunksz);

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(fullfile(save_path, 'dataset.h5'), batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(fullfile(save_path, 'dataset.h5'));

