function net = multitasknn_move(net, destination)
% multitasknn_move  Move a simple CNN between CPU and GPU
%    NET = multitasknn_move(NET, 'gpu') moves the network
%    on the current GPU device.
%    NET = multitasknn_move(NET, 'cpu') moves the network
%    on the CPU.

switch destination
    case 'gpu', moveop = @(x) gpuArray(x) ;
    case 'cpu', moveop = @(x) gather(x) ;
    otherwise, error('Unknown destination ''%s''.', destination) ;
end
for l = 1:numel(net.layers)
    switch net.layers{l}.type
        case {'conv', 'bnorm'}
            %             for f = {'filters', 'biases', 'filtersMomentum', 'biasesMomentum'}
            %                 f = char(f) ;
            %                 if isfield(net.layers{l}, f)
            %                     net.layers{l}.(f) = moveop(net.layers{l}.(f)) ;
            %                 end
            %             end
            for f = {'weights', 'momentum'}
                f = char(f) ;
                if isfield(net.layers{l}, f)
                    for j = 1:numel(net.layers{l}.(f))
                        net.layers{l}.(f){j} = moveop(net.layers{l}.(f){j}) ;
                    end
                end
            end
        otherwise
            % custom layer
    end
end