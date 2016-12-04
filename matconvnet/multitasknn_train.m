function net = multitasknn_train(net, imdbPackageDir, getBatch, varargin)
% MT_CNN_TRAIN   Demonstrates training a CNN on large train samples
% which are cached as several .mat data
%    MT_CNN_TRAIN() is an example learner implementing stochastic
%    gradient descent with momentum to train a CNN. It can be used
%    with different datasets and tasks by providing a suitable
%    getBatch function.
%
%    The function automatically restarts after each training epoch by
%    checkpointing.
%
%    The function supports training on CPU or ONE GPU
%    TODO: pre-load net package, coz loading a package need a moment

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
opts.batchSize = 200 ;
opts.train = [] ;
opts.val = [] ;
opts.numEpochs = 100 ;
opts.gpus = 1 ; % which GPU devices to use (none, one, or more)
opts.learningRate = 0.0001 ;
opts.continue = true ;
opts.expDir = fullfile('data','exp') ;
opts.conserveMemory = false ;
opts.backPropDepth = +inf ;
opts.sync = false ;
opts.cudnn = true ;
opts.weightDecay = 0.0001 ;
opts.momentum = 0.95 ;
opts.errorFunction = 'multiclass' ;
opts.errorLabels = {} ;
opts.plotDiagnostics = false ;
opts = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end

% -------------------------------------------------------------------------
%                                                               Preparation
% -------------------------------------------------------------------------

% get sample list
imdb_list = dir([imdbPackageDir '/' '*.mat']);
for i = 1 : numel(imdb_list)
    imdb_list(i).name = fullfile(imdbPackageDir, filesep, imdb_list(i).name);
end

% setup GPUs or CPU

    numGpus = 0;


% setup error calculation function
% notice: these error function DO NOT contribute anything in train process
% only provide some info how far the train process goes
if ischar(opts.errorFunction)
    switch opts.errorFunction
        case 'none'
            opts.errorFunction = @error_none ;
        case 'multiclass'
            opts.errorFunction = @error_multiclass ;
            if isempty(opts.errorLabels), opts.errorLabels = {'top1e', 'top5e'} ; end
        case 'binary'
            opts.errorFunction = @error_binary ;
            if isempty(opts.errorLabels), opts.errorLabels = {'bine'} ; end
        case 'euclidean'
            opts.errorFunction = @error_distance ;
            if isempty(opts.errorLabels), opts.errorLabels = {'max-dist', 'mean-dist'} ; end
        otherwise
            error('Uknown error function ''%s''', opts.errorFunction) ;
    end
end

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------
for epoch=1:opts.numEpochs
    learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
    
    % fast-forward to last checkpoint
    modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
    modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;
    if opts.continue
        if exist(modelPath(epoch),'file')
            if epoch == opts.numEpochs
                load(modelPath(epoch), 'net', 'info') ;
            end
            continue ;
        end
        if epoch > 1
            fprintf('resuming by loading epoch %d\n', epoch-1) ;
            load(modelPath(epoch-1), 'net', 'info') ;
        end
    end
    
    imdb_list = imdb_list(randperm(numel(imdb_list))); % shuffle packages
    stats.train = [];
    stats.val = [];
    for package = 1 : numel(imdb_list)
        % if only one package exist, load only once
        if numel(imdb_list) >= 2 || ~exist('imdb', 'var')
            fprintf('loading %dth(%d) imdb...', package, numel(imdb_list));
            imdb = load(imdb_list(package).name);
        end
        opts.train = find(imdb.images.set==1) ;
        opts.val = find(imdb.images.set==2) ;
        if isnan(opts.train), opts.train = [] ; end
        train = opts.train(randperm(numel(opts.train))) ; % shuffle
        val = opts.val ;
        if numGpus <= 1
            [net,stats_train] = process_package(imdb, train, getBatch, net, learningRate, epoch, numGpus, opts) ;
            [~,stats_val] = process_package(imdb, val, getBatch, net, 0, epoch, numGpus, opts) ;
        end
        % tracking all the states
        if package == 1
            stats.train =  stats_train;
            stats.val   =  stats_val;
        else
            stats.train = stats.train + stats_train;
            stats.val = stats.val + stats_val;
        end
    end
    % save
    stats.train(2:end) = stats.train(2:end) ./ numel(imdb_list);
    stats.val(2:end) = stats.val(2:end) ./ numel(imdb_list);
    sets = {'train', 'val'};
    legends = {};
    for f = sets
        f = char(f) ;
        n = numel(eval(f));
        info.(f).speed(epoch) = n / stats.(f)(1);
        for k = 1 : numel(stats.(f))-1
            info.(f).objective{k}(epoch) = stats.(f)(k+1);
            legends{end+1} = [f, '_', int2str(k)];
        end
    end
    save(modelPath(epoch), 'net', 'info');
    
    figure(1) ; clf ; 
    for k = 1 : numel(info.train.objective)
        semilogy(1:epoch, info.train.objective{k}, '.-', 'linewidth', 2) ;
        hold on ;
    end
    for k = 1 : numel(info.val.objective)
        semilogy(1:epoch, info.val.objective{k}, '.-', 'linewidth', 2) ;
        hold on ;
    end
    xlabel('training epoch') ; ylabel('energy') ;
    grid on ;
    h=legend(legends) ;
    set(h,'color','none');
    title('objective') ;
    drawnow ;
    print(1, modelFigPath, '-dpdf') ;
end
end

% -------------------------------------------------------------------------
function  [net_cpu, stats] = process_package(imdb, subset, getBatch, ...
    net_cpu, learningRate, epoch, numGpus, opts)
% -------------------------------------------------------------------------
% move CNN to GPU as needed
if numGpus == 1
    net = multitasknn_move(net_cpu, 'gpu') ;
    one = gpuArray(single(1)) ;
else
    net = net_cpu;
    one = single(1);
end

% validation mode if learning rate is zero
trainMode = learningRate > 0 ;
if trainMode, mode = 'training' ; else mode = 'validation' ; end
node_ends = net.dag.node_ends;
objs = zeros(numel(node_ends), 1);
stats = [] ;
batch_processed_num = 0;
sample_num = numel(subset);
for t = 1 : opts.batchSize : sample_num
    fprintf('%s: epoch %03d: batch %3d/%3d: ', mode, epoch, ...
        fix(t/opts.batchSize)+1, ceil(sample_num/opts.batchSize));
    batchSize = min(opts.batchSize, sample_num - t + 1) ;
    batchTime = tic ;
    numDone = 0 ;
    
    % get this image batch and prefetch the next
    batchEnd = min(t + opts.batchSize - 1, sample_num);
    batch = subset(t : batchEnd) ;
    data = getBatch(imdb, batch) ;
    
    if numGpus == 1 && iscell(data)
        for k = 1 : numel(data)
            data{k} = gpuArray(data{k});
        end
    elseif numGpus == 1
        data = gpuArray(data);
    end
    
    % FP & BP Net
    if trainMode, dzdy = one; else dzdy = [] ; end
    [res, gradients]= multitasknn(net, data, dzdy, ...
        'disableDropout', ~trainMode, ...
        'backPropDepth', opts.backPropDepth, ...
        'sync', opts.sync, ...
        'cudnn', opts.cudnn) ;
    % Update Net
    if trainMode
        net = multitasknn_update_net(net, gradients, batchSize, ...
            'learningRate', learningRate, ...
            'weightDecay', opts.weightDecay, ...
            'momentum', opts.momentum) ;
    end
    err = opts.errorFunction(data{2}, res) ./ batchSize;
    % get errors
    for k = 1 : numel(node_ends)
        objs(k) = gather(res(node_ends(k)).x);
    end
    numDone = numDone + numel(batch);
    batch_processed_num = batch_processed_num + 1;
    % print learning statistics for each mini-batch
    batchTime = toc(batchTime) ;
    stats = sum([stats,[batchTime; objs]],2); % works even when stats=[]
    speed = batchSize/batchTime ;
    
    fprintf(' lr:%.7g', learningRate);
    fprintf(' %.3f s (%.2f data/s)', batchTime, speed) ;
    for k = 1 : numel(objs)
        fprintf(' obj_%s: %.4g', ...
            net.dag.node_map_num2str(net.dag.node_ends(k)), objs(k)/batchSize) ;
    end
    fprintf(' top1: %.3f top5: %.3f', err) ;
    fprintf(' [%d/%d]', numDone, batchSize);
    fprintf('\n') ;
end

% stats: [total_time; average_errors]
stats(2:end) = stats(2:end) ./ batch_processed_num;

if numGpus == 1
    net_cpu = multitasknn_move(net, 'cpu') ;
else
    net_cpu = net ;
end
end

function err = error_multiclass(labels, res)
% -------------------------------------------------------------------------
predictions = gather(res(end-1).x) ;
[~,predictions] = sort(predictions, 3, 'descend') ;

% be resilient to badly formatted labels
if numel(labels) == size(predictions, 4)
  labels = reshape(labels,1,1,1,[]) ;
end

% skip null labels
mass = single(labels(:,:,1,:) > 0) ;
if size(labels,3) == 2
  % if there is a second channel in labels, used it as weights
  mass = mass .* labels(:,:,2,:) ;
  labels(:,:,2,:) = [] ;
end

error = ~bsxfun(@eq, predictions, labels) ;
err(1,1) = sum(sum(sum(mass .* error(:,:,1,:)))) ;
err(2,1) = sum(sum(sum(mass .* min(error(:,:,1:5,:),[],3)))) ;
end
