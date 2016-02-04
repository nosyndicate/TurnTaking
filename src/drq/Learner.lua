local LSTM = require 'LSTM'
local modelUtils = require 'ModelUtils'

local drql = torch.class('drq.Learner');


function drql:__init(args)

	self.inputSize = args.inputSize;
	self.rnnSize = args.rnnSize;
	self.actionSize = args.actionNum * args.actionNum; -- the output is the joint action
	
	self.networkParameters = nil;
	self.networkGradParameters = nil
	
	self.selfNetwork = self:createNetwork();
	self.opponentNetwork = self:createNetwork();

	self.maxLength = 0 -- the max length of the an episode
	
	self.history = {};
	
	-- parameters for the internal state of the LSTM
	self.initC = torch.zeros(1, args.rnnSize);
	self.initH = initC:clone();
	self.finalC = initC:clone();
	self.finalH = initC:clone();
	self.prevC = nil;
	self.prevH = nil;

end


function drql:createNetwork()
	local modules = {};
	
	modules.embed = nn.Sequential():add(nn.Linear(self.inputSize, self.rnnSize));
	-- TODO: let's see if we need to add a rectifier layer for non linearity
	modules.lstm = LSTM.create(self.rnnSize);
	modules.top = nn.Sequential():add(nn.Linear(self.rnnSize, self.actionSize));
	
	self.networkParameters, self.networkGradParameters = modelUtils.combineAllParameters(modules.embed, modules.lstm, modules.top);
	
	self.networkParameters:uniform(-0.08,0.08); -- random initialization
	
	return modules;
end


function drql:reset()
	-- need to zero out the history
	self.history = {};
end

function drql:learn()

end

function drql:step(state, reward, terminal)
	local stateInput = torch.Tensor();
	stateInput:cat(state, reward);  -- first construct the new state
	--[[
	local embeddings = {};
	local prevC = {[0] = initC};  -- set the initial state of c and h
	local prevH = {[0] = initH};  -- this is equal to say prevH[0] = initH
	local jointQ = {};
	

	embeddings[t] = clones.embed[t]:forward(x[{{}, t}])
		
	prevC[t],prevH[t] = unpack(clones.lstm[t]:forward({embeddings[t],prevC[t-1],prevH[t-1]})); -- input is table
		
	predictions[t] = clones.top[t]:forward(prevH[t]);
		
	local _,p = predictions[t]:max(1);
	print('prediction:'..p[1]..', target is '..y[t][1]);
		
	loss = loss + clones.criterion[t]:forward(predictions[t],y[t]); 
	end
	--]]
	
end


function drql:report()
	--print(self:formatWeightNorms(self.networkParameters));
	--print(self:formatWeightNorms(self.networkGradParameters));
end







