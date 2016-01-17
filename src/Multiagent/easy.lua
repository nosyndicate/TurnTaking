local easy = {};


easy.matrix = {{10,0,0},{0,0,0},{0,0,0}};
easy.numAction = 3; -- how many available action 
easy.numState = 1; -- how many state variable

function easy:playGame(actionOne, actionTwo)
	local reward = self.matrix[actionOne][actionTwo];
	return reward, reward;
end



return easy;