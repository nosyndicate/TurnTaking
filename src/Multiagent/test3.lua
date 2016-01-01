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

	local agent1 = rl.LenienceReinforce(model1, policy1, optimizer1);
	local agent2 = rl.LenienceReinforce(model2, policy2, optimizer2);
	agent1:setLearningRate(0.01);
	agent2:setLearningRate(0.01);

	local state = torch.Tensor({1});
	
	print(model1:forward(state));
	print(model2:forward(state));
	
	for i = 1,2000 do
		local average1, average2 = 0,0;
		-- repeat 10 trials
		for j = 1,100 do
			agent1:startTrial();
			agent2:startTrial();
			local r1, r2 = ro:playGame(agent1:getAction(state),agent2:getAction(state));
			agent1:step(state, r1);
			agent2:step(state, r2);
			agent1:endTrial();
			agent2:endTrial();
			average1 = average1 + r1;
			average2 = average2 + r2;
		end
		agent1:learn(nil, nil);
		agent2:learn(nil, nil);
		average1 = average1/100;
		average2 = average2/100;
		if i%50==0 then
			print("average is "..average1..","..average2);
			print(model1:forward(state));
			print(model2:forward(state));
		end
	end
	
	print(model1:forward(state));
	print(model2:forward(state));
	


end



main();







