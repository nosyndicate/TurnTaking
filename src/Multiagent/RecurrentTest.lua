require 'torch'
require 'nn'
require 'rl'
local LSTM = require 'LSTM'
local modelUtils = require 'ModelUtils'
local battle = require 'Multiagent.battle'

local cmd = torch:CmdLine();
cmd:text();
cmd:text('Options');

cmd:option('-iteration',6,'iteration to play game battle');
cmd:option('-threshold',0.0001,'the threshold for stopping');
cmd:option('-rnnSize',10,'how many cell in rnn layer');

local cmdParams = cmd:parse(arg);


function withInThreshold(new, old)
	if math.abs(new-old) < cmdParams.threshold then
		return true;
	else
		return false;
	end
end


function sample(model1, model2, state)
	local outputSeq1 = {};
	local outputSeq2 = {};
	local prev_c1 = torch.zeros(1, cmdParams.rnnSize);
	local prev_h1 = prev_c1:clone();
	local prev_c2 = torch.zeros(1, cmdParams.rnnSize);
	local prev_h2 = prev_c2:clone();
	for i=1,6 do
		local embedding1 = model1.input:forward(state);
    	local next_c1, next_h1 = unpack(model1.lstm:forward{embedding1, prev_c1, prev_h1})
    	local output1 = model1.output:forward(next_h1);
    	prev_c1:copy(next_c1)
    	prev_h1:copy(next_h1)
    	local embedding2 = model2.input:forward(state);
    	local next_c2, next_h2 = unpack(model2.lstm:forward{embedding2, prev_c2, prev_h2})
    	local output2 = model2.output:forward(next_h2);
    	prev_c1:copy(next_c2)
    	prev_h1:copy(next_h2)
    	
    	print("iteration"..i);
    	print(output1);
    	print(output2);
	end

end


function main()

	local model1 = {};
	local model2 = {};
	
	model1.input = nn.Sequential():add(nn.Linear(1, cmdParams.rnnSize));
	model1.lstm = LSTM.create(cmdParams.rnnSize);
	model1.output = nn.Sequential():add(nn.Linear(cmdParams.rnnSize, 2)):add(nn.SoftMax());
	
	model2.input = nn.Sequential():add(nn.Linear(1, cmdParams.rnnSize));
	model2.lstm = LSTM.create(cmdParams.rnnSize);
	model2.output = nn.Sequential():add(nn.Linear(cmdParams.rnnSize, 2)):add(nn.SoftMax());
	
	local policy1 = rl.DiscretePolicy(2);
	local policy2 = rl.DiscretePolicy(2);
	
	local optimizer1 = rl.StochasticGradientDescent(modelUtils.combineAllParameters(model1.input, model1.lstm, model1.output));
	local optimizer2 = rl.StochasticGradientDescent(modelUtils.combineAllParameters(model2.input, model2.lstm, model2.output));


	optimizer1.params:uniform(-0.08,0.08);
	optimizer2.params:uniform(-0.08,0.08);
	
	
	local agent1 = rl.RecurrentReinforce(model1, policy1, optimizer1, cmdParams.rnnSize, 6);
	local agent2 = rl.RecurrentReinforce(model2, policy2, optimizer2, cmdParams.rnnSize, 6);
	agent1:setLearningRate(0.003);
	agent2:setLearningRate(0.003);
	
	-- it is repeated game, we only have one state
	local state = torch.Tensor({1});
	
	sample(model1, model2, state);
	
	-- let's first try 2000 iterations
	for i = 1,2000 do
		-- let's do 50 trials per iterations
		for j = 1,50 do
			agent1:startTrial();
			agent2:startTrial();
			for k = 1, cmdParams.iteration do
				local a1,p1 = agent1:getAction(state);
				local a2,p2 = agent2:getAction(state);
				local r1, r2 = battle:playGame(a1,a2);
				agent1:step(state, r1, a2);
				agent2:step(state, r2, a1);
			end
			agent1:endTrial();
			agent2:endTrial();
		end
		
		agent1:learn(nil, nil);
		agent2:learn(nil, nil);
		
		if i%50==0 then
			sample(model1, model2, state);
		end
	end
	
	
end




main();