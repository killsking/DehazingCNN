classdef tf_loss_lse < dagnn.ElementWise
  %TF_LOSS_LSE Least Square Error for input 1 and input 2
  %   Typically, Input port 1: the prediction, 
  %   Input port 2: the ground truth label
  
  properties
    res; % [M,N]. residual
    sz;  % [d1,...,dK]. size for the data at input 1
  end
  
  methods
    function obj = tf_loss_lse()
      obj.i = [inputs{1}, inputs{2}];
      obj.o = inputs{1};
    end % tf_loss_lse
    
    function outputs = forward(obj, inputs, params)
      % the prediction and target 
      ob.sz  = size( obj.i(1).a );
      pre = obj.i(1).a;
      tar = reshape(obj.i(2).a, obj.sz);
      % the residual
      obj.res = pre - tar;
      obj.res = reshape(obj.res, ...
        [prod(ob.sz(1:end-1)), obj.sz(end)] ); 
      % the loss
      obj.o.a = 0.5 * sum( (obj.res).^2, 1 ); 
    end % fprop
    
    function ob = bprop(ob)
      % just using the "cache", keep it the size with .i
      obj.i(1).d = reshape(obj.res, obj.sz );
    end % bprop
    
  end
  
end


