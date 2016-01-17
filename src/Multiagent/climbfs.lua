local climbfs = {};

climbfs.matrix = {{{10,12},{5,-65},{8,-8}},
				{{5,-65},{14,0},{12,0}},
				{{5,-5},{5,-5},{10,0}}};
climbfs.numAction = 3; -- how many available action 
climbfs.numState = 1; -- how many state variable

function climbfs:playGame(actionOne, actionTwo)
	local rewards = self.matrix[actionOne][actionTwo];
	local value = torch.uniform(); -- sample a value from [0,1)
	if value < 0.5 then
		return rewards[1], rewards[1];
	else
		return rewards[2], rewards[1];
	end
end



return climbfs;