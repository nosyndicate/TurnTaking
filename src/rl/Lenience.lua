require 'torch'

local Lenience, parent = torch.class('rl.Lenience','rl.PolicySearch');


function Lenience:__init(model, actor, optimizer, temperature)
	parent.__init(self, model, actor, optimizer);
	self.trials = {};
	self.rewardCurrentTrial = nil;
	self.gradientCurrentTrial = nil;
	self.temperature = temperature;
	self.previousAverageReward = -math.huge; -- negative infinity
	self.decay = 0.9999;
end

function Lenience:step(s, r)
	
	self.optimizer.grads:zero();

	-- first accumulate the reward
	-- TODO: this means we are not discount the reward, may consider as future work
	self.rewardCurrentTrial = self.rewardCurrentTrial + r;
	
	-- then compute the gradient of current step and add to gradient of current trial
	local dLogPolicyDOutput = self.actor:backward();

	self.model:backward(s, dLogPolicyDOutput);
	
	self.gradientCurrentTrial:add(self.optimizer.grads);
end

function Lenience:startTrial()
	-- initialize the reward and gradient
	self.rewardCurrentTrial = 0;
	self.gradientCurrentTrial = self.optimizer.grads:clone():zero();
end

function Lenience:endTrial()
	-- put the reward and gradient into the corresponding table for learning
	table.insert(self.trials,{gradient = self.gradientCurrentTrial, reward = self.rewardCurrentTrial});
end





function Lenience:optimisticTrialsEstimation()
	-- first get the number of trials
	local l = #self.trials;
	-- create the gradient estimator
	local gradientEstimator = self.trials[1].gradient:clone():zero();
	local counter = 0;
	local sum = 0;
	-- go through the better trials and make the estimation
	for i = 1,l do
		local r = self.trials[i].reward;
		
		if r > self.previousAverageReward then
			sum = sum + r;
			local reward = torch.Tensor(gradientEstimator:size()):fill(r);
			-- multiple the accumulated gradient for each trial with the reward
			local tempGradient = torch.cmul(self.trials[i].gradient, reward);
			-- sum up the trials
			gradientEstimator:add(tempGradient);
			counter = counter + 1;
		end
	end
	
	local temp = nil;
	if counter == 0 then
		temp = 0;
	else
		temp = sum/counter;
	end
		
	-- return the average of the gradient estimator
	return torch.div(gradientEstimator, counter), temp; 
end

-- estimate the gradient based on all trials
function Lenience:averageTrialsEstimation()

	-- first get the number of trials
	local l = #self.trials;
	local sum = 0;
	-- create the gradient estimator
	local gradientEstimator = self.trials[1].gradient:clone():zero();
	
	-- go through the all trials and make the estimation
	for i = 1,l do
		sum = sum + self.trials[i].reward;
		local reward = torch.Tensor(gradientEstimator:size()):fill(self.trials[i].reward);
		
		-- multiple the accumulated gradient for each trial with the reward
		local tempGradient = torch.cmul(self.trials[i].gradient, reward);
		-- sum up the trials
		gradientEstimator:add(tempGradient);
	end
		
	-- return the average of the gradient estimator
	return torch.div(gradientEstimator, l), sum/l; 
end

-- episodic Reinforce only learns at the end of trials
function Lenience:calculateGradient(s, r)
	
	-- based on the prob, determine use which way to update gradient
	local prob = 1 - math.exp(-1/self.temperature);
	local gradient, sum = nil, nil;
	if torch.uniform() < prob then
		print("average");
		gradient, sum = self:averageTrialsEstimation();
	else
		print("optimistic");
		gradient, sum = self:optimisticTrialsEstimation();
	end
	-- update the temperature
	self.temperature = self.temperature * self.decay;
	
	-- compute the average payoff with given policy as future reference
	if self.previousAverageReward < sum then
		self.previousAverageReward = sum;
	end
		
	print("previous average reward is "..self.previousAverageReward);
	-- reset the trials
	self.trials = {};

	return gradient;
end



