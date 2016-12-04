function net = multitasknn_preprocess_net(net, varargin)
% MULTITASKNN_PREPROCESS_NET will fill empty fields and build DAG graph for
% both data layers and parameter layers

% net = multitasknn_preprocess_net(net, 'show', 1) will plot graphs.

opts.node_start = 'data';
opts.show = true;
opts = vl_argparse(opts, varargin) ;

net = fill_empty_fields(net);
[dag, node_map, node_map_num2str, nodes_unique]= build_dag_for_nodes(net);
net.dag.dag = dag;
net.dag.node_map_str2num = node_map;
net.dag.node_map_num2str = node_map_num2str;
% dag ends
net.dag.node_start = node_map(opts.node_start);
net.dag.node_ends = find_dag_ends(dag, net.dag.node_start);

% topological sorting dag
net.dag.topo_list = topo_sort(net.dag.dag, net.dag.node_start);
net.dag.topo_list_reverse = flip(net.dag.topo_list);
% net.dag.topo_list_reverse = topo_sort(net.dag.dag', net.dag.node_ends);

[dag_layer, layer_map, layer_map_num2str, layers_unique] = build_dag_for_layers(net, dag, node_map);

net.dag.dag_layer = dag_layer;
net.dag.layer_map = layer_map;
net.dag.layer_map_num2str = layer_map_num2str;
% view dag
if opts.show
    % view(biograph(dag));
    view(biograph(dag, nodes_unique));
    view(biograph(dag_layer, layers_unique));
    fprintf('pausing...\n');
    pause;
end

end

function net = fill_empty_fields(net)
  for i=1:numel(net.layers)
    if ~isfield(net.layers{i}, 'type')
        error('Layer %d has no type field', i);
    end
    if ~isfield(net.layers{i}, 'top') || ~isfield(net.layers{i}, 'bottom')
        error('Layer %d has no [bottom|top] field', i);
    end
    if ~isfield(net.layers{i}, 'name')
        net.layers{i}.name = horzcat(net.layers{i}.type, int2str(i));
        warning('Layer %d has no name field, program will automatically assign name %s', ...
            i, net.layers{i}.name);
    end
    % weight
    if isfield(net.layers{i}, 'weights')
      J = numel(net.layers{i}.weights) ;
      for j=1:J
        net.layers{i}.momentum{j} = zeros(size(net.layers{i}.weights{j}), 'single') ;
      end
      if ~isfield(net.layers{i}, 'learningRate')
        net.layers{i}.learningRate = ones(1, J, 'single') ;
      end
      if ~isfield(net.layers{i}, 'weightDecay')
        net.layers{i}.weightDecay = ones(1, J, 'single') ;
      end
    end
  end
end
function [dag, node_map, node_map_num2str, nodes_unique]= build_dag_for_nodes(net)
nodes = {};
for i = 1 : numel(net.layers)
    layer = net.layers{i};
    if  iscell(layer.bottom)
        for j = 1 : numel(layer.bottom)
            nodes{end+1} = layer.bottom{j};
        end
    else
        nodes{end+1} = layer.bottom;
    end
    if  iscell(layer.top)
        for j = 1 : numel(layer.top)
            nodes{end+1} = layer.top{j};
        end
    else
        nodes{end+1} = layer.top;
    end
end

% [nodes_unique, idx, idx_] = unique(nodes);
% node_id = [1:node_num];
% node_map = containers.Map(nodes_unique, node_id);
% node_map_num2str = containers.Map(node_id, nodes_unique);
node_map = containers.Map;
cnt = 1;
for i = 1 : numel(nodes)
    node = nodes{i};
    if isKey(node_map, node)
        continue
    else
        node_map(node) = cnt;
        cnt = cnt + 1;
    end
end
nodes_unique = keys(node_map);
node_id = values(node_map);
node_map_num2str = containers.Map(node_id, nodes_unique);
nodes_unique = values(node_map_num2str);

node_num = numel(nodes_unique);
dag = zeros(node_num, node_num);
for i = 1 : numel(net.layers)
    layer = net.layers{i};
    if  iscell(layer.bottom)
        for j = 1 : numel(layer.bottom)
            from = node_map(layer.bottom{j});
            to   = node_map(layer.top);
            dag(from, to) = i;
        end
        continue;
    end
    if  iscell(layer.top)
        for j = 1 : numel(layer.top)
            to = node_map(layer.top{j});
            from = node_map(layer.bottom);
            dag(from, to) = i;
        end
        continue;
    end
    if  ~iscell(layer.top) && ~iscell(layer.bottom)
            to = node_map(layer.top);
            from = node_map(layer.bottom);
            dag(from, to) = i;
    end
end

end
function [node_ends] = find_dag_ends(dag, node_start)
node_visited = zeros(size(dag, 1), 1);
node = node_start;
queue{1} = node;
node_ends = [];
while numel(queue) > 0
    node = queue{1};
    % disp(node);
    queue(1) = [];
    out = dag(node, :);
    if sum(out) <= 0
        node_ends(end+1) = node;
    else
        out_idx = find(out > 0);
        for i = 1 : numel(out_idx)
            if ~node_visited(out_idx(i))
                queue{end+1} = out_idx(i);
                node_visited(out_idx(i)) = 1;
            end
        end
    end
end
end
function [topo_list] = topo_sort(dag, node_start)
topo_list = [];
if iscell(node_start)
    node_start = horzcat(node_start{:});
end
node_zero_indegree = node_start;
while numel(node_zero_indegree) > 0
    n = node_zero_indegree(1);
    node_zero_indegree(1) = [];
    topo_list(end+1) = n;
    out = find(dag(n, :) > 0);
    for i = 1 : numel(out)
        m = out(i);
        dag(n, m) = 0;
        if sum (dag(:, m)) <= 0 % incoming of m is none
            node_zero_indegree(end+1) = m;
        end
    end
end
if sum(dag(:)) > 0
    topo_list = [];
    warning('this network is not a DAG')
end
end
function [dag_layer, layer_map, layer_map_num2str, layers_unique] = build_dag_for_layers(net, dag_node, node_map)
dag_layer = zeros(numel(net.layers), numel(net.layers));
layer_str = {};
for i = 1 : numel(net.layers)
    if isfield( net.layers{i}, 'name')
        layer_str{end+1} = net.layers{i}.name;
    else
        layer_str{end+1} = horzcat(net.layers{i}.type, int2str(i));
    end
end

layer_id = [1: numel(net.layers)];
layer_map = containers.Map(layer_str, layer_id);
layer_map_num2str = containers.Map(layer_id, layer_str);
layers_unique = values(layer_map_num2str);

for i = 1 : numel(net.layers)
    froms = {};
    tos = {};
    layer = net.layers{i};
    if  iscell(layer.bottom)
        for j = 1 : numel(layer.bottom)
            from = node_map(layer.bottom{j});
            froms{end+1} = from;
        end
        continue;
    end
    if  iscell(layer.top)
        for j = 1 : numel(layer.top)
            to = node_map(layer.top{j});
            tos{end+1} = to;
        end
        continue;
    end
    if  ~iscell(layer.top) && ~iscell(layer.bottom)
            tos{1} = node_map(layer.top);
            froms{1} = node_map(layer.bottom);
    end
    for j = 1 : numel(froms)
        for k = 1 : numel(tos)
            f = froms{j};
            t = tos{k};
            l = dag_node(f, t);
            ffs = find(dag_node(:, f) > 0);
            tts = find(dag_node(t, :) > 0);
            for q = 1 : numel(ffs)
                ff = ffs(q);
                lf = dag_node(ff, f);
                dag_layer(lf, l) = ff;
            end
            for q = 1 : numel(tts)
                tt = tts(q);
                lt = dag_node(t, tt);
                dag_layer(l, lt) = tt;
            end
        end
    end
end
end