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
	local optimizer1 = rl.StochasticGradientDescent(model1:getParameters());
	local optimizer2 = rl.StochasticGradientDescent(model2:getParameters());

	
	local agent1 = rl.GPOMDP(model1, policy1, optimizer1);
	local agent2 = rl.GPOMDP(model2, policy2, optimizer2);

	local state = torch.Tensor({1});
	
	print(model1:forward(state));
	print(model2:forward(state));
	
	for i = 1,2000*50 do
		local r1, r2 = ro:playGame(agent1:getAction(state),agent2:getAction(state));
		agent1:learn(state, r1);
		agent2:learn(state, r2);
		if i%50==0 then
			print("reward is "..r1..","..r2);
		end
	end
	
	print(model1:forward(state));
	print(model2:forward(state));
	


end



main();







