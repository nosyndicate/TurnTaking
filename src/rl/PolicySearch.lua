require 'torch'

local PolicySearch, parent = torch.class('rl.PolicySearch','rl.Learner');


function PolicySearch:__init(model, actor)
	parent.__init(self, model);
	self.actor = actor;
	-- the output of the model, which is also the parameter of the distribution for stochastic policy
	self.output = nil;
end

function PolicySearch:getAction(s)
	-- get the parameters for the distribution of the stochastic policy
	local output = self.model:forward(s);
	-- sample from the distribution 
	local action = self.actor:getAction(output);
	print(action);
	return action;
end

function PolicySearch:learning(s, r)
	local gradient = calculateGradient();
	
end

function PolicySearch:calculateGradient()
	
end






