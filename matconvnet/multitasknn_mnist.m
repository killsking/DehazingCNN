% test multitasknn no mnist
function net = multitasknn_mnist()

%------------------------------------------------------
%                                              Init net
%------------------------------------------------------
%net = multitasknn_mnist_init_net();
net = multitask_mnist_init_simple_net() ;
net = multitasknn_preprocess_net(net, 'show', 1);

%------------------------------------------------------
%                                                 Data 
%------------------------------------------------------
dataDir = fullfile('data','mnist') ;
expDir = fullfile('data','mnist-baseline') ;
imdbPath = fullfile(expDir, 'imdb.mat');
if exist(imdbPath, 'file')
    imdbPackageDir = fullfile(dataDir,'imdbDir');
   if ~exist(imdbPackageDir, 'dir')
       fprintf('make dir %s\n', imdbPackageDir);
       mkdir(imdbPackageDir);
       fprintf('copy %s to %s\n', imdbPath, imdbPackageDir);
       copyfile(imdbPath, imdbPackageDir);
   end
else
  error('You can run cnn_mnist.m to get mnist imdb.mat file');
end

%------------------------------------------------------
%                                                Train 
%------------------------------------------------------
net = multitasknn_train(net, imdbPackageDir, @getBatch, ...
    'numEpochs', 10, 'expDir', expDir);
end

function data = getBatch(imdb, batch)
data{1}=imdb.images.data(:,:,:,batch);
data{2}=imdb.images.labels(:,batch);
 data{3}=data{2};
end