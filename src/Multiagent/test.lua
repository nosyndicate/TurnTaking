require 'torch'
require 'nn'
require 'rl'
local ro = require 'Multiagent.ro'






function main()
	-- the model, input is just a single number as the state number
	-- than we do a linearly transformation and then output three values and squash them into a distribution
	local model1 = nn.Sequential():add(nn.Linear(1, 3)):add(nn.SoftMax());
	local policy1 = rl.DiscretePolicy(3);
	local model2 = nn.Sequential():add(nn.Linear(1, 3)):add(nn.SoftMax());
	local policy2 = rl.DiscretePolicy(3);
	
	local agent1 = rl.Reinforce(model1, policy1);
	local agent2 = rl.Reinforce(model2, policy2);
	
	for i = 1,10 do
		local r1, r2 = ro:playGame(agent1:getAction(torch.Tensor({1})),agent2:getAction(torch.Tensor({1})));
		agent1:step(torch.Tensor({1}), r1);
		agent2:step(torch.Tensor({1}), r2);
	end
	
	
	
	
	
end



main();







