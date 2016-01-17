local climbps = {};

-- using the same strategy for climb fs
climbps.matrix = {{{10,10},{-30,-30},{0,0}},
				{{-30,-30},{14,0},{6,6}},
				{{0,0},{0,0},{5,5}}};
climbps.numAction = 3; -- how many available action 
climbps.numState = 1; -- how many state variable

function climbps:playGame(actionOne, actionTwo)
	local rewards = self.matrix[actionOne][actionTwo];
	local value = torch.uniform(); -- sample a value from [0,1)
	if value < 0.5 then
		return rewards[1], rewards[1];
	else
		return rewards[2], rewards[2];
	end
end



return climbps;