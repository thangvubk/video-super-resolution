clear; close all;

%% Configurationi
% NOTE: you can modify this part
test_set = 'vid4/walk'; %(myanmar, India)
scale = 3;

%% Create save path for high resolution and low resolution images based on config
% NOTE: you should NOT modify the following parts
disp(sprintf('%10s: %s', 'Test set', test_set));
disp(sprintf('%10s: %d', 'Scale', scale));

scale_dir = strcat(int2str(scale), 'x');

% example
% read_path = '../data/test/myanmar/'
% save_path = '../preprocessed_data//test/myanmar/3x/'
read_path = fullfile('../data', 'test', test_set);
save_path = fullfile('../preprocessed_data', 'test', test_set, scale_dir);

if exist(save_path, 'dir') ~= 7
	mkdir(save_path)
end

is_init_data = true;

% get folder in read_path
dirs = dir(read_path);


count = 0;
for i_dir = 1 : length(dirs)
    scene_dir = dirs(i_dir).name;
	if scene_dir(1) ~= 's' %valid folder begin with 's'
        continue
    end

    disp(sprintf('processing dir: %s', scene_dir));

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
        lr_im = imresize(lr_im ,[hei, wid],'bicubic');
        
		if is_init_data
			data = zeros(hei, wid, 5, 1);
			label = zeros(hei, wid, 5, 1);
			is_init_data = false;
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

