function x2 = concat(x1, x2, dzdy, varargin)
if nargin == 2 || isempty(dzdy)
    if numel(x2) == 0
        x2 = x1;
    else
        assert(size(x1,1)==size(x2,1) && size(x1,2)== size(x2,2)&&size(x1,4)==size(x2,4))
        x2 = cat(3, x2, x1);
    end
else
    opts.precedes = [];
    opts.scope = [];
    opts.pre =  [];
    opts = vl_argparse(opts, varargin);
    for i = 1 : numel(opts.precedes)
        if opts.pre == opts.precedes(i)
            scope = opts.scope(i, :);
        end
    end
    x2 = dzdy(:,:,scope(1):scope(2),:);
end
end