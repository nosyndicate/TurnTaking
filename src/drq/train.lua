if not dqn then
	require 'initenv'
end


local cmd = torch.CmdLine();

cmd:text();
cmd:text('Option:');
cmd:option('-episodes', 10^3, 'number of training episodes to perform');
cmd:Option('-game', 'simplebattle', 'which game to play');

cmd:text();

local opt = cmd:parse(arg);

local completeEpisodes = 0;



local agent1 = agentSetup();
local agent2 = agentSetup(); 


-- play an certain episodes number
while completeEpisodes < opt.episode do
	
	local state, reward, terminal = game:newGame();
	
	-- start a new terminal
	while not terminal do
		local action1 = agent1:step(state, reward, terminal);
		local action2 = agent2:step(state, reward, terminal);
	
		-- if game is not ended, keep going
		if not terminal then
			state, reward, terminal = game:play(action1, action2);
		end
	end
	
	-- increase the counter
	completeEpisodes = completeEpisodes + 1;
	
	agent1:learn();
	agent2:learn();
	
	if completeEpisodes%1000==0 then
		collectgarbage();
	end
	
end


function agentSetup()
	local agent = Learner();
	
	return agent;
end





