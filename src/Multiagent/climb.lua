local climb = {};

climb.matrix = {{10,-30,0},{-30,7,6},{0,0,5}};
climb.numAction = 3; -- how many available action 
climb.numState = 1; -- how many state variable

function climb:playGame(actionOne, actionTwo)
	local reward = self.matrix[actionOne][actionTwo];
	return reward, reward;
end



return climb;