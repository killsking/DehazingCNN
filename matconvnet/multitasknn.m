function [res, gradients] = multitasknn(net, data, dzdy, varargin)
% MULTITASKNN  Evaluates a multi-output CNN (DAG)
%   RES = MULTITASKNN(NET, X) evaluates the convnet NET on data X.
%   RES = MULTITASKNN(NET, X, DZDY) evaluates the convnent NET and its
%   derivative on data X and output derivative DZDY.
%   net.layers
%   net.dag
%   data
%      cell, data{1} input
%      cell, data{2:end} label1, label2, ...
opts.sync = false ;
opts.disableDropout = false ;
opts.freezeDropout = false ;
opts.cudnn = false ;
opts.backPropDepth = +inf ;

opts = vl_argparse(opts, varargin);
if (nargin <= 2) || isempty(dzdy) || ~iscell(data) || numel(data)<=1
    doder = false ; % val & test mode
else
    doder = true ;
end

if opts.cudnn
    cudnn = {'CuDNN'} ;
else
    cudnn = {'NoCuDNN'} ;
end
if iscell(data) 
    gpuMode = isa(data{1}, 'gpuArray') ;
else
    gpuMode = isa(data, 'gpuArray') ;
end

dag = net.dag;
node_num = size(dag.dag, 1);
layer_num = numel(net.layers);

res = struct(...
    'x', cell(1,node_num), ...
    'dzdx', cell(1,node_num), ...
    'aux', cell(1,node_num), ...
    'in',  num2cell(zeros(1,node_num)), ... % in-degree of the current node
    'out', num2cell(zeros(1,node_num)), ... % out-degree of the current node
    'time', num2cell(zeros(1,node_num)), ...
    'backwardTime', num2cell(zeros(1,node_num))) ;
gradients = struct(...
    'dzdw', cell(1,layer_num)) ;

res(1).x = data ;

% -------------------------------------------------------------------------
%                                                           Forward Network
% -------------------------------------------------------------------------
topo_list = dag.topo_list;
node_start = dag.node_start;
assert(numel(topo_list) == node_num);
assert(numel(node_start) == 1);
assert(node_start == topo_list(1));
for i = 1 : numel(topo_list)
    node = topo_list(i);
    succeeds  = find(dag.dag(node, :) > 0);
    res(node).out = numel(succeeds);
    if ~any(succeeds)
        continue;
    end
    layer = net.layers{dag.dag(node, succeeds(1))};
    % for data layer 
    % todo: 
    if strcmp(layer.type, 'data')
        n = numel(succeeds);
        res(node).in = 0;
        if ~iscell(res(node).x)
            res(succeeds(1)).x = res(node).x;
        elseif ~numel(res(node).x) == n
        else
            for j = 1 : n
                res(succeeds(j)).in = res(succeeds(j)).in + 1;
                res(succeeds(j)).x = res(node).x{j};
                if j >= 2
                    res(succeeds(j)).aux.is_label = true;
                end
            end
        end
        continue;
    end
    % for normal layer, including loss, concat, conv, pool, norm, relu, dropout
    res(node).time = tic;
    for j = 1 : numel(succeeds)
        suc = succeeds(j);
        l = dag.dag(node, suc);
        res(suc).in = res(suc).in + 1;
        res(suc) = forward_layer(res(node), res(suc), net.layers{l}, node, cudnn, opts);
        if gpuMode && opts.sync
            wait(gpuDevice) ;
        end
    end
    res(node).time = toc(res(node).time) ;
end
% -------------------------------------------------------------------------
%                                                          Backward Network
% -------------------------------------------------------------------------
if doder
    for i = 1 : numel(dag.node_ends)
        res(dag.node_ends(i)).dzdx = dzdy(min(numel(dzdy), i));
    end
    topo_list_reverse = dag.topo_list_reverse;
    for i = 1 : numel(topo_list_reverse)
        node = topo_list_reverse(i);
        res(node).backwardTime = tic;
        precedes = find(dag.dag(:, node) > 0);
        for j = 1 : numel(precedes)
            pre = precedes(j);
            % fprintf('<%d, %d>\n', [node, pre]);
            l = dag.dag(pre, node);
            [res(pre), gradients(l).dzdw]= backward_layer(res(node), res(pre), net.layers{l}, pre,  ...
                cudnn, opts);
        end
        if gpuMode && opts.sync
            wait(gpuDevice) ;
        end
        res(node).backwardTime = toc(res(node).backwardTime) ;
    end
end
end

function res_suc = forward_layer(res_node, res_suc, layer, node_pre, cudnn, opts)
switch layer.type
    case 'conv'
        res_suc.x = vl_nnconv(res_node.x, layer.weights{1}, layer.weights{2}, ...
            'pad', layer.pad, 'stride', layer.stride, ...
            cudnn{:}) ;
    case 'convt'
        res_suc.x = vl_nnconvt(res_node.x, layer.weights{1}, layer.weights{2}, ...
            'crop', layer.crop, 'upsample', layer.upsample, ...
            cudnn{:}) ;
    case 'pool'
        res_suc.x = vl_nnpool(res_node.x, layer.pool, ...
            'pad', layer.pad, 'stride', layer.stride, ...
            'method', layer.method, ...
            cudnn{:}) ;
    case 'normalize'
        res_suc.x = vl_nnnormalize(res_node.x, layer.param) ;
    case 'softmax'
        res_suc.x = vl_nnsoftmax(res_node.x) ;
    case 'relu'
        res_suc.x = vl_nnrelu(res_node.x) ;
    case 'sigmoid'
        res_suc.x = vl_nnsigmoid(res_node.x) ;
    case 'noffset'
        res_suc.x = vl_nnnoffset(res_node.x, layer.param) ;
    case 'spnorm'
        res_suc.x = vl_nnspnorm(res_node.x, layer.param) ;
    case 'dropout'
        if opts.disableDropout
            res_suc.x = res_node.x ;
        elseif opts.freezeDropout
            [res_suc.x, res_suc.aux] = vl_nndropout(res_node.x, ...
                'rate', layer.rate, 'mask', res_suc.aux) ;
        else
            [res_suc.x, res_suc.aux] = vl_nndropout(res_node.x, ...
                'rate', layer.rate) ;
        end
    case 'bnorm'
        res_suc.x = vl_nnbnorm(res_node.x, layer.weights{1}, layer.weights{2});
    case 'pdist'
        res_suc = vl_nnpdist(res_node.x, layer.p, 'noRoot', ...
            layer.noRoot, 'epsilon', layer.epsilon) ;
        % modified layers
    case 'concat'
        if ~isfield(res_suc.aux, 'precedes')
            res_suc.aux.precedes = node_pre;
            res_suc.aux.scope  = [1, size(res_node.x, 3)];
        else
            res_suc.aux.precedes = horzcat(res_suc.aux.precedes, node_pre);
            res_suc.aux.scope  = [res_suc.aux.scope;...
                size(res_suc.x, 3)+1, size(res_suc.x, 3)+size(res_node.x, 3)...
                ];
        end
        res_suc.x = concat(res_node.x, res_suc.x) ;
    case 'euclideanloss'
        if isfield(res_node.aux, 'is_label') && res_node.aux.is_label
            res_suc.aux = res_node.x;
        elseif ~isempty(res_suc.aux)
            res_suc.x = euclideanloss(res_node.x, res_suc.aux) ;
        end
    case {'loss', 'softmaxlog'}
        % make sure program always visit label node before any other node 
        % connected to loss layer.
        if isfield(res_node.aux, 'is_label') && res_node.aux.is_label
            res_suc.aux = res_node.x;
        elseif ~isempty(res_suc.aux) 
            res_suc.x = vl_nnloss(res_node.x, res_suc.aux) ;
        end
        % Deprecated: use `vl_nnloss` instead
    case 'softmaxloss'
        if isfield(res_node.aux, 'is_label') && res_node.aux.is_label
            res_suc.aux = res_node.x;
        elseif ~isempty(res_suc.aux) 
            res_suc.x = vl_nnsoftmaxloss(res_node.x, res_suc.aux) ;
        end
    otherwise
        error('Unknown layer type [%s]', layer.type);
end
end

function [res_pre, dzdw]= backward_layer(res_node, res_pre, layer, pre, cudnn, opts)
dzdx = [];
dzdw = cell(1,2);
switch layer.type
    case 'conv'
        [dzdx, dzdw{1}, dzdw{2}] = ...
            vl_nnconv(res_pre.x, layer.weights{1}, layer.weights{2}, ...
            res_node.dzdx, ...
            'pad', layer.pad, 'stride', layer.stride, ...
            cudnn{:}) ;
        
    case 'convt'
        [dzdx, dzdw{1}, dzdw{2}] = ...
            vl_nnconvt(res_pre.x, layer.weights{1}, layer.weights{2}, ...
            res_node.dzdx, ...
            'crop', layer.crop, 'upsample', layer.upsample, ...
            cudnn{:}) ;
        
    case 'pool'
        dzdx = vl_nnpool(res_pre.x, layer.pool, res_node.dzdx, ...
            'pad', layer.pad, 'stride', layer.stride, ...
            'method', layer.method, ...
            cudnn{:}) ;
        
    case 'normalize'
        dzdx = vl_nnnormalize(res_pre.x, layer.param, res_node.dzdx) ;
        
    case 'softmax'
        dzdx = vl_nnsoftmax(res_pre.x, res_node.dzdx) ;
        
    case 'relu'
        if ~isempty(res_pre.x)
            dzdx = vl_nnrelu(res_pre.x, res_node.dzdx) ;
        else
            dzdx = vl_nnrelu(res_node.x, res_node.dzdx) ;
        end
        
    case 'sigmoid'
        dzdx = vl_nnsigmoid(res_pre.x, res_node.dzdx) ;
    case 'noffset'
        dzdx = vl_nnnoffset(res_pre.x, layer.param, res_node.dzdx) ;
    case 'spnorm'
        dzdx = vl_nnspnorm(res_pre.x, layer.param, res_node.dzdx) ;
    case 'dropout'
        if opts.disableDropout
            dzdx = res_node.dzdx ;
        else
            dzdx = vl_nndropout(res_pre.x, res_node.dzdx, ...
                'mask', res_node.aux) ;
        end
    case 'bnorm'
        [dzdx, dzdw{1}, dzdw{2}] = ...
            vl_nnbnorm(res_pre.x, layer.weights{1}, layer.weights{2}, ...
            res_node.dzdx) ;
    case 'pdist'
        dzdx = vl_nnpdist(res_pre.x, layer.p, res_node.dzdx, ...
            'noRoot', layer.noRoot, 'epsilon', layer.epsilon) ;
    case 'custom'
        res_pre = layer.backward(layer, res_pre, res_node) ;
    case 'data'
    % modified layers
    case 'concat'
        dzdx = concat([], [], res_node.dzdx, ...
            'pre', pre, ...
            'scope', res_node.aux.scope, 'precedes', res_node.aux.precedes);
    case  {'loss', 'softmaxlog'}
        if ~(isfield(res_pre.aux, 'is_label') && res_pre.aux.is_label)
            % loss node stores the label
            res_pre.dzdx = vl_nnloss(res_pre.x, res_node.aux, res_node.dzdx) ;
        end
    case 'euclideanloss'
        if ~(isfield(res_pre.aux, 'is_label') && res_pre.aux.is_label)
            res_pre.dzdx = euclideanloss(res_pre.x, res_node.aux, res_node.dzdx);
        end
        % Deprecated: use `vl_nnloss` instead
    case 'softmaxloss'
        if ~(isfield(res_pre.aux, 'is_label') && res_pre.aux.is_label)
            % loss node stores the label
            res_pre.dzdx = vl_nnsoftmaxloss(res_pre.x, res_node.aux, res_node.dzdx);
        end
end
if ~isempty(dzdx)
    r = single(min(1 / res_pre.out, 1));
    if isempty(res_pre.dzdx)
        res_pre.dzdx = r .* dzdx ;
    else
        res_pre.dzdx = res_pre.dzdx + r .* dzdx;
    end
end
end
