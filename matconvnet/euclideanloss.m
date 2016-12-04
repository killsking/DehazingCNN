function Y = euclideanloss(X, c, dzdy)
%EUCLIDEANLOSS Summary of this function goes here
%   Detailed explanation goes here

assert(numel(X) == numel(c));
[d1,d2,d3,d4] = size(X);
c = single(reshape(c, d1,d2,d3,d4));

if nargin == 2 || (nargin == 3 && isempty(dzdy))
    % Y DO NOT need to be divided by d(4)
    Y = 1 / (2*d4*d3) * sum(subsref((X - c) .^ 2, substruct('()', {':'})));
    % Y = 1 / d4 *  sum(subsref((X - c) .^ 2, substruct('()', {':'})));

elseif nargin == 3 && ~isempty(dzdy)
    % mask = c > mean(c(:));
    assert(numel(dzdy) == 1);
    % Y DO NOT need to be divided by d(4) 
    Y = (dzdy * (X - c)); 
end
end