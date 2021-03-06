function net = multitask_mnist_init_simple_net(varargin)
% CNN_MNIST_LENET Initialize a CNN similar for MNIST
opts.useBnorm = true ;
opts = vl_argparse(opts, varargin) ;

rng('default');
rng(0) ;

f=1/100 ;
net.layers = {} ;
net.layers{end+1} = struct(...
    'name', 'data', ...
    'bottom', 'data', ...
    'top', {{'input', 'label1'}}, ...
    'type', 'data' ...
    ) ;
net.layers{end+1} = struct('type', 'conv', ...
    'name', 'covn1', ...
    'bottom', 'input',...
    'top', 'conv1', ...
    'weights', {{f*randn(5,5,1,20, 'single'), zeros(1, 20, 'single')}}, ...
    'stride', 1, ...
    'pad', 0) ;

net.layers{end+1} = struct('type', 'conv', ...
    'name', 'covn2', ...
    'bottom', 'covn1',...
    'top', 'conv2', ...
    'weights', {{f*randn(3,3,20,50, 'single'),zeros(1,50,'single')}}, ...
    'stride', 1, ...
    'pad', 1) ;

net.layers{end+1} = struct('type', 'conv', ...
    'name', 'covn3', ...
    'bottom', 'covn2',...
    'top', 'conv3', ...
    'weights', {{f*randn(5,5,50,1, 'single'),  zeros(1,1,'single')}}, ...
    'stride', 1, ...
    'pad', 2) ;

net.layers{end+1} = struct('type', 'euclideanloss', ...
        'name', 'loss1', ...
    'bottom', {{'conv3', 'label1'}},...
    'top', 'loss1') ;



