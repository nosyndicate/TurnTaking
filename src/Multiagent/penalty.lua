local penalty = {};

penalty.matrix = {{10,0,-10},{0,0,0},{-10,0,10}};
penalty.numAction = 3; -- how many available action 
penalty.numState = 1; -- how many state variable

function penalty:playGame(actionOne, actionTwo)
	local reward = self.matrix[actionOne][actionTwo];
	return reward, reward;
end



return penalty;