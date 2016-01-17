require 'torch'
local modelUtils = require 'ModelUtils'

local RecurrentReinforce, parent = torch.class('rl.RecurrentReinforce','rl.PolicySearch');

-- length indicate the length we do the BPTT
function RecurrentReinforce:__init(model, actor, optimizer, rnnSize, length)
	parent.__init(self, model, actor, optimizer);
	self.rnnSize = rnnSize;
	self.trials = {};
	-- history restore all the stuff we need in BPTT
	self.history = {};
	self.t = 1;
	self.length = length;
	self.clones = {};
	self.rewardCurrentTrial = nil;
	self.gradientCurrentTrial = nil;
	for name, submodule in pairs(self.model) do
		--print('cloneing '..name);
		--print(self.length);
		self.clones[name] = modelUtils.cloneManyTimes(submodule, self.length);
	end
end

function RecurrentReinforce:getAction(s)
	-- get the parameters for the distribution of the stochastic policy
	local embedding = self.clones.input[self.t]:forward(s);
	table.insert(self.history.embedding, embedding);
	table.insert(self.history.input, s);

	local c, h = unpack(self.clones.lstm[self.t]:forward({embedding,self.history.c[self.t-1],self.history.h[self.t-1]})); -- input is table
	table.insert(self.history.c, c);
	table.insert(self.history.h, h);


	local parameters = self.clones.output[self.t]:forward(h);
	-- sample from the distribution 
	local action = self.actor:getAction(parameters);
	return action, parameters;
end

function RecurrentReinforce:step(s, r)
	self.optimizer.grads:zero();

	-- instead of taking the derivative with respect to the state
	-- we need to take it with respect with all the history
	
	self.rewardCurrentTrial = self.rewardCurrentTrial + r;
	
	-- then compute the gradient of current step and add to gradient of current trial
	local dLogPolicyDOutput = self.actor:backward();
	
	-- we need to do BPTT (may be truncated) in here
	-- Usually, we have to clone the net for multiple times (sequence length) , since according to the standard of the torch module.lua
	-- each module need to have their own state variable gradInput (gradient w.r.t. input) and output (forward pass after given the input),
	-- thus if we don't do the clone and use the same module over and over again, we would have overwrite the output (in forward pass) 
	-- and gradInput (in backward pass)
	
	-- In reinforcement learning, since we cannot determine the sequence length in advance in very large scale game, we have to truncated in some way
	-- that is we clone the net for some fix length l (BPTT truncated length), and if the trajectory length is within the l, we don't need to overwrite 
	-- output state variable in the clones, otherwise, we have to use the clones as circular queue.
	
	-- In REINFORCE, each step we need to compute the gradient w.r.t. the whole history until current steps, good things is that since we already 
	-- accumulate the gradient in the previous steps, we can overwrite the gradInput of the previous steps, thus clone the net to an array is enough,
	-- we don't need to clone for each steps
	
	-- NOTE: start to do back propagate through time here
	-- we only care about the gradient the last output (action) w.r.t. the previous things
	local dEmbedding = {};
	-- c value is used by next step, since we don't care at future at bptt, we will set it as zeros
	local dPrevC = {[self.t] = torch.zeros(1, self.rnnSize);}; 
	local dPrevH = {};
	
	for i = self.t, 1, -1 do
		dPrevH[i] = self.clones.output[i]:backward(self.history.h[i], dLogPolicyDOutput);
		
		dEmbedding[i], dPrevC[i-1], dPrevH[i-1] = unpack(self.clones.lstm[i]:backward({self.history.embedding[i], self.history.c[i-1], self.history.h[i-1]},{dPrevC[i], dPrevH[i]}))
	
		self.clones.input[i]:backward(self.history.input[i], dEmbedding[i]);
	end
	
	-- than we need to accumulate the gradient
	self.gradientCurrentTrial:add(self.optimizer.grads);
	
	-- increase the time step counter
	self.t = self.t + 1;
	
end


-- episodic Reinforce only learns at the end of trials
function RecurrentReinforce:calculateGradient(s, r)
	-- first get the number of trials
	local l = #self.trials;
	-- create the gradient estimator
	local gradientEstimator = self.trials[1].gradient:clone():zero();
	
	for i = 1,l do
		local reward = torch.Tensor(gradientEstimator:size()):fill(self.trials[i].reward);
		-- multiple the accumulated gradient for each trial with the reward
		local tempGradient = torch.cmul(self.trials[i].gradient, reward);
		-- sum up the trials
		gradientEstimator:add(tempGradient);
	end
	
	-- return the average of the gradient estimator
	return torch.div(gradientEstimator, l); 
end



function RecurrentReinforce:startTrial()
	self.rewardCurrentTrial = 0;
	self.gradientCurrentTrial = self.optimizer.grads:clone():zero();
	-- initialize the lstmState and time index
	self.t = 1;  -- the first step is 1
	local initC = torch.zeros(1, self.rnnSize);
	local initH = torch.zeros(1, self.rnnSize);
	-- we need to initialize the c and h in index 0
	self.history = {c={[0]=initC}, h={[0]=initH}, embedding={}, input={}};
end

function RecurrentReinforce:endTrial()
	table.insert(self.trials,{gradient = self.gradientCurrentTrial, reward = self.rewardCurrentTrial});
	
	-- NOTE:clamp the range of the gradient
	self.optimizer.grads:clamp(-5,5);
end





