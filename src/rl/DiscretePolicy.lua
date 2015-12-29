require 'torch'


--[[
	A sampler sample from a multinomial distribution to produce the action
--]]

local DiscretePolicy = torch.class('rl.DiscretePolicy');


function DiscretePolicy:__init(actNum)
	self.actNum = actNum;
	self.action = torch.Tensor();
end


function DiscretePolicy:getAction(distribution)
	assert(self.actNum == distribution:size()[1], 'mismatch of policy distribution');
	
	-- sample 1 time without replacement from the distribution
	local index = torch.multinomial(distribution, 1); 
	
	-- convert the index into one-hot representation
	self.action = distribution:clone();
	-- set all element to zero
	self.action:zero();
	-- put 1 at the index along the first dimension (row)
	-- encode in one-hot encoding
	self.action:scatter(1, index, 1);
	
	-- return the index in number	
	return self.action:nonzero()[1][1];
end