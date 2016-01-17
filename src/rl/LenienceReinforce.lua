require 'torch'

local LenienceReinforce, parent = torch.class('rl.LenienceReinforce','rl.PolicySearch');


function LenienceReinforce:__init(model, actor, optimizer, useOptimalBaseline)
	parent.__init(self, model, actor, optimizer);
	self.reward = {};
	self.gradient = {};
	self.rewardCurrentTrial = nil;
	self.gradientCurrentTrial = nil;
	-- the default optimal baseline is turned off
	self.useOptimalBaseline = useOptimalBaseline or false;
end

function LenienceReinforce:step(s, r)
	if r<8 then
		return;
	end

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

function LenienceReinforce:startTrial()
	-- initialize the reward and gradient
	self.rewardCurrentTrial = 0;
	self.gradientCurrentTrial = self.optimizer.grads:clone():zero();
end

function LenienceReinforce:endTrial()
	-- put the reward and gradient into the corresponding table for learning
	table.insert(self.reward, self.rewardCurrentTrial);
	table.insert(self.gradient, self.gradientCurrentTrial);
end


-- episodic Reinforce only learns at the end of trials
function LenienceReinforce:calculateGradient(s, r)
	-- first get the number of trials
	local l = #self.reward;
	-- create the gradient estimator
	local gradientEstimator = self.gradient[1]:clone():zero();
	
	-- default optimal baseline is 0
	local optimalBL = gradientEstimator:clone():zero();
	if self.useOptimalBaseline then
		optimalBL = self:optimalBaseline();
	end
	
	for i = 1,l do
		local reward = torch.Tensor(gradientEstimator:size()):fill(self.reward[i]);
		if self.useOptimalBaseline then
			r:csub(optimalBL);
		end
		-- multiple the accumulated gradient for each trial with the reward
		local tempGradient = torch.cmul(self.gradient[i], reward);
		-- sum up the trials
		gradientEstimator:add(tempGradient);
	end
	
	-- TODO: may need to consider to add optimal baseline here
	
	-- reset trials
	self.reward = {};
	self.gradient = {};
	
	-- return the average of the gradient estimator
	return torch.div(gradientEstimator, l); 
end

function LenienceReinforce:optimalBaseline()
	local l = #self.reward;
	local numerator = self.gradient[1]:clone():zero();
	local denominator = numerator:clone();
	for i = 1,l do
		local innerProduct = torch.cmul(self.gradient[i], self.gradient[i]);
		numerator:add(torch.mul(innerProduct, self.reward[i]));
		denominator:add(innerProduct);
	end
	local optimalBL = torch.cdiv(numerator, denominator);
	
	return optimalBL;
end


