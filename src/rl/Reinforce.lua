require 'torch'

local Reinforce, parent = torch.class('rl.Reinforce','rl.PolicySearch');


function Reinforce:__init(model, actor, optimizer)
	parent.__init(self, model, actor, optimizer);
	self.reward = {};
	self.gradient = {};
	self.rewardCurrentTrial = nil;
	self.gradientCurrentTrial = nil;
end

function Reinforce:step(s, r)
	self.optimizer.grads:zero();

	-- first accumulate the reward
	-- TODO: this means we are not discount the reward, may consider as future work
	self.rewardCurrentTrial = self.rewardCurrentTrial + r;
	
	-- then compute the gradient of current step and add to gradient of current trial
	local dLogPolicyDOutput = self.actor:backward();
	--print("log policy is ");
	--print(dLogPolicyDOutput);
	self.model:backward(s, dLogPolicyDOutput);
	--print("grads is ");
	--print(self.optimizer.grads);
	
	self.gradientCurrentTrial:add(self.optimizer.grads);
end

function Reinforce:startTrial()
	-- initialize the reward and gradient
	self.rewardCurrentTrial = 0;
	self.gradientCurrentTrial = self.optimizer.grads:clone():zero();
end

function Reinforce:endTrial()
	-- put the reward and gradient into the corresponding table for learning
	table.insert(self.reward, self.rewardCurrentTrial);
	table.insert(self.gradient, self.gradientCurrentTrial);
end


-- episodic Reinforce only learns at the end of trials
function Reinforce:calculateGradient()
	-- first get the number of trials
	local l = #self.reward;
	-- create the gradient estimator
	local gradientEstimator = self.gradient[1]:clone():zero();
	for i = 1,l do
		-- multiple the accumulated gradient for each trial with the reward
		local tempGradient = torch.mul(self.gradient[i], self.reward[i]);
		-- sum up the trials
		gradientEstimator:add(tempGradient);
	end
	
	-- TODO: may need to consider to add optimal baseline here
	
	-- return the average of the gradient estimator
	return torch.div(gradientEstimator, l); 
end


