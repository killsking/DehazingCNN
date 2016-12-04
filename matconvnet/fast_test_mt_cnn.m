% test multitasknn no mnist
%------------------------------------------------------
%                                              Init net
%------------------------------------------------------
%net = multitasknn_mnist_init_net2();
 net = multitasknn_mnist_init_net();  % original mnist net
net = multitasknn_preprocess_net(net, 'show', 1);

%------------------------------------------------------
%                                                 Data 
%------------------------------------------------------
n = 20;
data{1} = rand(28, 28, 1, n, 'single');
data{2} = rand(28, 28, 1, n, 'single');
for i = 1 : n
    data{1}(:,:,:,i)  = data{1}(:,:,:,i) .* (i/10);
    data{2}(:,:,:,i)  = data{1}(:,:,:,i) .* (0);
end
data{3} = data{2};
%------------------------------------------------------
%                                                Train 
%------------------------------------------------------

for i = 1 : 10
    fprintf('%d ', i);
    [res, gradients] = multitasknn(net, data, 1);
    net = multitasknn_update_net(net, gradients, n, ...
        'learningRate', 2e-2, 'weightDecay', 1e-6);
    fprintf('%.8f\n', res(end).x);
end

%------------------------------------------------------
%                                                  Test 
%------------------------------------------------------
[res_1, gradients] = multitasknn(net, data, 1);
