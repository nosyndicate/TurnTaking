require 'torch'

local Reinforce, parent = torch.class('rl.Reinforce','rl.PolicySearch');


function Reinforce:__init(model, actor)
	parent.__init(self, model, actor);
	self.reward = {};
	self.gradient = {};
	self.rewardCurrentTrial = nil;
	self.gradientCurrentTrial = nil;
end

function Reinforce:step(s, r)
	-- first accumulate the reward
	-- TODO: this means we are not discount the reward, may consider as future work
	self.rewardCurrentTrial = self.rewardCurrentTrial + r;
	
	-- then compute the gradient of current step
	
	
end

function Reinforce:startTrial()
	-- initialize the reward and gradient
	self.rewardCurrentTrial = 0;
	local _, g = model:getParameters();
	self.gradientCurrentTrial = torch.Tensor(g:size()):zeros();
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
	local gradientEstimator = torch.Tensor(self.gradient[1]:size()):zeros();
	for i = 1,l do
		-- multiple the accumulated gradient for each trial with the reward
		local tempGradient = torch.mul(self.gradient[i], reward[i]);
		-- sum up the trials
		gradientEstimator = torch.add(tempGradient, gradientEstimator);
	end
	
	-- TODO: may need to consider to add optimal baseline here
	
	
	-- return the average of the gradient estimator
	return torch.div(gradientEstimator, l); 
end


