clear all;clc;
% set path for training samples and test samples directory
addpath('large_scale_svm');
addpath('sift');
addpath(genpath('sparse_coding'));
% directory setup for training samples
img_dir = 'image';% directory for subfolder images
data_dir = 'data';                  % directory to save the sift features of the chosen subfolder
subfolder = 'CMVR';


%% parameter setting
% sift descriptor extraction
skip_cal_sift = true;                % if 'skip_cal_sift' is true, skip extracting sift feature patches for training
gridSpacing = 128;
patchSize = 256;
nrml_threshold = 1;                  % low contrast region normalization threshold (descriptor length)
% dictionary training for sparse coding
skip_dic_training = false;           % if 'skip_dic_training' is true, skip dictionary training
nBases = 1024;
nsmp = 40000;                      % the total patches
beta = 1e-5;                        % a small regularization for stablizing sparse coding
num_iters = 1;                     % the iteration times. why 50? it takes too much time.
% feature pooling parameters
pyramid = [1, 2, 4];                % spatial block number on each level of the pyramid
gamma = 0.15;
knn = 200;                          % find the k-nearest neighbors for approximate sparse coding
% ScSPM for training sets.
skip_cal_sparse = false;
skip_svm = false;
%%============================================ Codebook=====================================================================%

cmvr_img_dir = fullfile(img_dir, subfolder);
cmvr_data_dir = fullfile(data_dir, subfolder);
%%=================================================================================================================%



%==================calculate sift features or retrieve the database directory==============%
if skip_cal_sift
    database = retr_database_dir(cmvr_data_dir);
else
    [database, lenStat, patchNumberToatal, patchNumberPerImage] = CalculateSiftDescriptor(cmvr_img_dir, cmvr_data_dir, gridSpacing, patchSize, nrml_threshold);
    save('log.mat', 'patchNumberToatal', 'patchNumberPerImage');
    %[database,lenStat] = CalculateLBPSiftDescriptor(cmvr_img_dir, cmvr_data_dir, gridSpacing, patchSize, maxImSize, mapping,nrml_threshold);
end
%=====================================================================================%
disp("===============================================");
%===========================% for exclude three persons' images out for cross-validation========================%
% randperm(n,k) returns a row vector containing k unique integers selected randomly from 1 to n inclusive.
testNumRate = 0.2;
total_num_test_image = 0;
fprintf("database.nclass: %d\n", database.nclass)
for ii = 1 : database.nclass
    Test(ii) = floor(database.imnum(ii) * testNumRate);
    total_num_test_image = total_num_test_image + Test(ii);
%     fprintf("%f\n", database.imnum(ii))
%     fprintf("%f\n", Test(ii))
%     fprintf("=======\n")
end

num_img = sum(database.imnum) - total_num_test_image;
fprintf("image number: %f\n", database.imnum)
fprintf("total_num_test_image: %f\n", total_num_test_image)
%==============================train the k-dimension(B) base for sparse coding=======================================%
Bpath = ['dictionary/dict_' subfolder '_' num2str(nBases) '.mat'];
if skip_dic_training
    load(Bpath);
else
    X = rand_sampling(database, nsmp, Test, num_img);
    [B, S, stat] = reg_sparse_coding(X, nBases, eye(nBases), beta, gamma, num_iters);
    save(Bpath, 'B', 'S', 'stat');
end
nBases = size(B, 2);% size of the dictionary
%==================================================================================================%




%======================================== calculate the sparse coding feature==========================================%
Spath=['dictionary/Sparse_' subfolder '_' num2str(nBases) '.mat'];
if skip_cal_sparse
    load(Spath);
else
    dimFea = sum(nBases*pyramid.^2);
    sc_fea = zeros(dimFea, num_img);
    sc_label = zeros(num_img, 1);

    disp('==================================================');
    fprintf('Calculating the sparse coding feature...\n');
    fprintf('Regularization parameter: %f\n', gamma);
    disp('==================================================');
    iter1 = 0;
    for i = 1 : database.nclass
        for m = Test(i) + 1 : database.imnum(i)
            iter1 = iter1+1;
            if ~mod(iter1, 50)
                fprintf('.\n');
            else
                fprintf('.');
            end
            fpath = database.path{i, m};
            load(fpath);
            if knn
                sc_fea(:, iter1) = sc_approx_pooling(feaSet, B, pyramid, gamma, knn);
            else
                sc_fea(:, iter1) = sc_pooling(feaSet, B, pyramid, gamma);
            end
            sc_label(iter1) = feaSet.label;
        end
    end
    save(Spath, 'sc_label', 'sc_fea');
end
%===============================================================================================================%




%============================= train SVM using sparse code and label of each image=============================================%

Wpath=['dictionary/svm.mat'];
if skip_svm
    load(Wpath);
else
    lambda = 0.1;                       % regularization parameter for w
    [w, b, class_name] = li2nsvm_multiclass_lbfgs(sc_fea', sc_label, lambda);
    save(Wpath,'w','b','class_name');
end
%===============================================================================================================%





%===================================validation for the test imageSet picked up earlier==================================================%
for l = 1 : database.nclass
    P{l} = database.path{l, 1};
    index = strfind(P{l}, '/');
    fprintf('P{l}: %s\n',P{l});
    P{l} = P{l}(index(2)+1:index(3)-1); % AMD/AMD1, ...
end

for ii=1:database.nclass
    Results=zeros(2,1);
    for jj = 1 : Test(ii)
        fpath = database.path{ii, jj};
        fprintf('Test Sample: %s\n',fpath);
        load(fpath);
%         I = imread(imgpath);
%         % calculate sift features
%         [feaSet_test,lenStat_test]=CalculateSiftDescriptor_Test(I, gridSpacing, patchSize, nrml_threshold);
%         %[feaSet_test]=CalculateLBPSiftDescriptor_Test(I, gridSpacing, patchSize,mapping,nrml_threshold);
        % calculate the sparse coding feature
        if knn
            sc_fea_test = sc_approx_pooling(feaSet, B, pyramid, gamma, knn);
        else
            sc_fea_test = sc_pooling(feaSet, B, pyramid, gamma);
        end


        [C, Y] = li2nsvm_multiclass_fwd(sc_fea_test', w, b, class_name);
        if C==1
            result='neg';
            Results(1)=Results(1)+1;
        else
            result='pos';
            Results(2)=Results(2)+1;
        end
        fprintf('Result: %s\n',result);
    end
    
    fprintf('---------------\n%s result: neg:%d, pos:%d \n', P{ii}, Results(1), Results(2));
end
%============================================================================================================================%
save('CMVR_result.mat');
