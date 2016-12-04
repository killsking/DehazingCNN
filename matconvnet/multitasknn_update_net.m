function net_cpu = multitasknn_update_net(net_cpu, gradients, batchSize, varargin)
opts.learningRate = 1e-3;
opts.weightDecay  = 1e-5;
opts.momentum     = 0.95;
opts = vl_argparse(opts, varargin);
lr_base = opts.learningRate;
for l = 1 : numel(net_cpu.layers)
    for j = 1:numel(gradients(l).dzdw)
        if isfield(net_cpu.layers{l}, 'weights')
            decay = opts.weightDecay * net_cpu.layers{l}.weightDecay(j);
            lr = lr_base * net_cpu.layers{l}.learningRate(j);
            net_cpu.layers{l}.momentum{j} = ...
                opts.momentum * net_cpu.layers{l}.momentum{j} ...
                - decay * net_cpu.layers{l}.weights{j} ...
                - (1 / batchSize) * gradients(l).dzdw{j} ;
            net_cpu.layers{l}.weights{j} = ...
                net_cpu.layers{l}.weights{j} + lr * net_cpu.layers{l}.momentum{j} ;
        end
    end
end
end