require 'torch'

local PolicySearch, parent = torch.class('rl.PolicySearch','rl.Learner');


function PolicySearch:__init(model, actor, optimizer)
	parent.__init(self, model);
	self.actor = actor;
	self.optimizer = optimizer;
	self.optimizer:setLearningRate(self.alpha);
end

function PolicySearch:getAction(s)
	-- get the parameters for the distribution of the stochastic policy
	local parameters = self.model:forward(s);
	-- sample from the distribution 
	local action = self.actor:getAction(parameters);
	return action;
end

function PolicySearch:learn(s, r)
	local gradient = self:calculateGradient();
	
	-- update the parameters with the gradient
	self.optimizer:gradientAscent(gradient);
end

function PolicySearch:calculateGradient()
	
end






