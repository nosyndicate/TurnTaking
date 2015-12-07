require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
local reber = require 'reber'
local LSTM = require 'LSTM'
local modelUtils = require 'ModelUtils';

local params = nil;
local modules = {};
local clones = {};
local parameters = nil;
local gradParameters = nil;
local sequenceMaxLength = nil;

local initC = nil;
local initH = nil;
local finalC = nil;
local finalH = nil;


local i = nil;   -- data x
local o = nil;   -- data y
local index = 1;



function feval(_p)
	collectgarbage();
	
	if _p ~= parameters then 
		parameters:copy(_p);
	end
	gradParameters:zero();
	
	-- get the mini batches --
	local x = i[index];
	local y = o[index];
	
	-- get the size of the column (2nd dimension) which is the sequence length of current tuple
	local seqLength = x:size(2);
	
	-- update index --
	index = index+1;
	if index > #i then
		index = 1;
	end
	
	-- forward pass --
	local embeddings = {};
	local prevC = {[0] = initC};  -- set the initial state of c and h
	local prevH = {[0] = initH};  -- this is equal to say prevH[0] = initH
	local predictions = {};
	local loss = 0;
	
	for t = 1, seqLength do
		embeddings[t] = clones.bottom[t]:forward(x[{{}, t}])
		
		prevC[t],prevH[t] = unpack(clones.lstm[t]:forward({embeddings[t],prevC[t-1],prevH[t-1]})); -- input is table
		
		predictions[t] = clones.top[t]:forward(prevH[t]);
		--print(predictions[t]);
		--print(y[t]);
		
		loss = loss + clones.criterion[t]:forward(predictions[t],y[t]); 
	end
	
	
	

	-- backward pass --
	local dEmbed = {};   -- d loss / d input 
	local dPrevC = {[seqLength] = finalC};  
	local dPrevH = {};
	
	for t = seqLength,1,-1 do     -- going backwards
		-- backprop through loss, and softmax/linear
		local dOutputAtT = clones.criterion[t]:backward(predictions[t],y[t]);
		

		-- Two cases for dloss/dh_t: 
        --   1. h_T is only used once, sent to the softmax (but not to the next LSTM timestep).
        --   2. h_t is used twice, for the softmax and for the next step. To obey the
        --      multivariate chain rule, we add them.
        if t == seqLength then
            assert(dPrevH[t] == nil)
            dPrevH[t] = clones.top[t]:backward(prevH[t], dOutputAtT);
        else
            dPrevH[t]:add(clones.top[t]:backward(prevH[t], dOutputAtT));
        end
		
		--   [gradInput] backward(input, gradOutput)
		dEmbed[t], dPrevC[t-1], dPrevH[t-1] = unpack(clones.lstm[t]:backward({x[{{},t}],prevC[t-1],prevH[t-1]},{dPrevC[t],dPrevH[t]}));
		
		clones.bottom[t]:backward(x[{{}, t}], dEmbed[t])
	end
	
	-- transfer final state to initial state (BPTT), may not need to be done accoding to char-rnn
	initC:copy(prevC[#prevC]);
	initH:copy(prevH[#prevH]);
	
	
	-- Alex Graves suggests clipping gradients to [-1,1] range or [-10,10] range during BPTT. otherwise, the gradient will explode
	gradParameters:clamp(-5,5);
	
	
	return loss, gradParameters;
	
end







local function train()
	local losses = {};
	local optimState = {
		learningRate = 1e-1,
    	weightDecay = 0,
    	momentum = 0,
    	learningRateDecay = 1e-7
	};  
	local iteration = 100000;
	for i = 1, iteration do
		local _, loss = optim.adagrad(feval, parameters, optimState);
		losses[#losses+1] = loss[1];
		
		if i % params.printIteration == 0 then
        	print(string.format("iteration %4d, loss = %6.8f, gradnorm = %6.4e", i, loss[1], gradParameters:norm()));
    	end
	end
end


local function test()
	
end 


local function main()
	local cmd = torch:CmdLine();
	cmd:text();
	cmd:text('Options');
	cmd:option('-dataSize',10000,'how many training sample to give');
	cmd:option('-rnnSize',10,'size of LSTM internal state');
	cmd:option('-epoch',1,'number of full passes through the training data');
	cmd:option('-batchSize',100,'number of training samples in each batch');
	cmd:option('-printIteration',100,'number of full passes through the training data');
	
	
	params = cmd:parse(arg);
	
	-- prepare data
	print('generating training samples');
	i,o,sequenceMaxLength = reber:getTrainingSamples(params.dataSize, 7);
	
	
	print('building model......');
	modules.bottom = nn.Sequential():add(nn.Linear(7, params.rnnSize)); -- convert the input from 7 into 10 internal states
	modules.lstm = LSTM.create(params.rnnSize);
	modules.top = nn.Sequential():add(nn.Linear(params.rnnSize, 7)):add(nn.LogSoftMax());
	modules.criterion = nn.ClassNLLCriterion();
	
	parameters, gradParameters = modelUtils.combineAllParameters(modules.lstm, modules.top);
	
	parameters:uniform(-0.08,0.08); -- random initialization
	
	-- cloned RNNs share the same parameters of the original RNN, but the only differences is that they can have their own output/gradInputs
	for name, submodule in pairs(modules) do
		print('cloneing '..name);
		clones[name] = modelUtils.cloneManyTimes(submodule, sequenceMaxLength, not submodule.parameters);
	end

	initC = torch.zeros(1, params.rnnSize);
	initH = initC:clone();
	finalC = initC:clone();
	finalH = initC:clone();

	
	train();
	
	test();

end


main();