require 'torch'
require 'nn'

local Learner = require 'Learner'
local battle = require 'battle'



local function playGame(playerOne, playerTwo, times)
	for i = 1,times do
		local actionOne = playerOne:getAction(i);
		local actionTwo = playerTwo:getAction(i);
		local rewardOne, rewardTwo = battle:playGame(actionOne, actionTwo);
		
		playerOne.learning(rewardOne, i);
		playerTwo.learning(rewardTwo, i);
	end
end


local function main()

	-- parse the arguments
	cmd = torch.CmdLine();
	cmd:text('Training recurrent policy for turn taking');
	cmd:text();
	cmd:text('Options');
	cmd:option('-seed',1234,'initial random seed');
	cmd:option('-rnnSize',10,'internal state of the lstm');
	cmd:option('-gameLength',10,'how many times the game should play');
	cmd:text();

	params = cmd:parse(arg);

	
	local playerOne = Learner:new({id = 1, rnnSize = params.rnnSize, inputSize = battle.numState, outputSize = battle.numAction});
	local playerTwo = Learner:new({id = 2, rnnSize = params.rnnSize, inputSize = battle.numState, outputSize = battle.numAction});
	
	-- start to play the game
	playGame(playerOne, playerTwo, params.gameLength);
end



main();
