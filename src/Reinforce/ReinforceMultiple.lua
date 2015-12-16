--[[
	This serves as the last layer in the neural network for producing the discrete action to take.
	Specifically, this module is used to produce the multinomial action in one hot encoding e.g. (0 0 0 1 0 ... 0)
	See section 6 of paper : Simple statistical gradient-following algorithms for connectionist reinforcement learning
	This code heavily use the code from dpnn : https://github.com/nicholas-leonard/dpnn/blob/master/ReinforceCategorical.lua
--]]

local ReinforceMultiple, parent = torch.class("ReinforceMultiple","nn.Module");

--[[
	g(y,p_1,p_2,p_3,...,p_n) - Multinomial distribution function 
	y      - Sampled values (1 to n) (self.output will encoding the sampled value in one-hot format)
	p_i    - Probability of sampling a value of i
	g(y,p_1,p_2,p_3,...,p_n) = p_i   (if y = i)
	       
	Input is the vector of p_i (p_1, p_2, p_3, ..., p_n)
	Output is the sampled value from the Multinomial distribution in one-hot encoding
--]]
function ReinforceBinary:updateOutput(input)
	self.output.resizeAs(input);
	self.sampleValue = self.sampleValue or input.new();
	self.sampleValue:resizeAs(input):copy(input):add()
	
	
	-- create one hot encoding
	self.output:zero();
	


	return self.output;
end

--[[
	Ignore the gradOutput since we are not measure the derivative of the loss w.r.t p
	Instead, we measure the derivative of log Multinomial w.r.t p
	p - probability vector (p_1, p_2, p_3, ..., p_n)
	  d ln(g(y,p))           1            d g(y,p)
	---------------- =  -----------  *  ------------
	      d p              g(y,p)           d p
	    
	=    1/p_i * 1          (if y = i)
	=    0                  (if y = 0)
--]]
function ReinforceBinary:updateGradInput(input, gradOutput)
	self.gradInput:resizeAs(input):zero();
	self.gradInput:copy(self.output);
	self.denominator = self.denominator or input.new();
	
	-- add some tiny value in case some probability is 0
	self.denominator:resizeAs(input):copy(input):add(0.00000001);
	self.gradInput:cdiv(self.denominator);

	
	-- multiply by reward, so the rest of the layer doesn't need to worry about the reward
	self.gradInput:cmul(self:rewardAs(input)); 
	
	-- multiplay by -1 (we are doing gradient ascent actually to maximize reward instead of minimizing the error)
	self.gradInput:mul(-1);
	
	return self.gradInput;
	
end