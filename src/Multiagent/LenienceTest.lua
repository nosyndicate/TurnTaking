require 'torch'
require 'nn'
require 'rl'
local climb = require 'Multiagent.climb'
local climbps = require 'Multiagent.climbps'
local climbfs = require 'Multiagent.climbfs'
local penalty = require 'Multiagent.penalty'
local cmd = torch:CmdLine();
cmd:text();
cmd:text('Options');
cmd:option('-game','climbps','which game to play');
cmd:option('-threshold',0.0001,'the threshold for stopping');


	
local cmdParams = cmd:parse(arg);
local game = nil;
if cmdParams.game == "climb" then
	game = climb;
elseif cmdParams.game == "climbps" then
	game = climbps;
elseif cmdParams.game == "climbfs" then
	game = climbfs;
else 
	game = penalty;
end

function withInThreshold(new, old)
	if math.abs(new-old) < cmdParams.threshold then
		return true;
	else
		return false;
	end
end


function main()
	-- the model, input is just a single number as the state number
	-- than we do a linearly transformation and then output three values and squash them into a distribution
	local model1 = nn.Sequential():add(nn.Linear(1, 3)):add(nn.SoftMax());
	local policy1 = rl.DiscretePolicy(3);
	local model2 = nn.Sequential():add(nn.Linear(1, 3)):add(nn.SoftMax());
	local policy2 = rl.DiscretePolicy(3);
	local optimizer1 = rl.StochasticGradientDescent(model1:getParameters());
	local optimizer2 = rl.StochasticGradientDescent(model2:getParameters());

	local agent1 = rl.Lenience(model1, policy1, optimizer1, 50);
	local agent2 = rl.Lenience(model2, policy2, optimizer2, 50);
	agent1:setLearningRate(0.0001);
	agent2:setLearningRate(0.0001);

	-- it is repeated game, we only have one state
	local state = torch.Tensor({1});

	print(model1:forward(state));
	print(model2:forward(state));

	local grad1Norm = 0;
	local grad2Norm = 0;
	local iteration = 0;

	for i=1,3000 do
		iteration = iteration + 1;

		-- we do 100 trial every time we want to learn
		for j = 1,100 do
			agent1:startTrial();
			agent2:startTrial();
			local r1, r2 = game:playGame(agent1:getAction(state),agent2:getAction(state));
			agent1:step(state, r1);
			agent2:step(state, r2);
			agent1:endTrial();
			agent2:endTrial();
		end
		agent1:learn(nil, nil);
		agent2:learn(nil, nil);

		local currentGrad1Norm = optimizer1.grads:norm();
		local currentGrad2Norm = optimizer2.grads:norm();

		if iteration%50==0 then
			print("the norm of gradient is "..currentGrad1Norm.." and "..currentGrad2Norm);
			print(model1:forward(state));
			print(model2:forward(state));
		end


	end

	print(model1:forward(state));
	print(model2:forward(state));
end

main();
