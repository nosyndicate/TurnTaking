local ro = {};

ro.matrix = {{10,0,0},{0,0,0},{0,0,0}};

ro.numAction = 3; -- how many available action 
ro.numState = 1; -- how many state variable

function ro:playGame(actionOne, actionTwo)
	local reward = self.matrix[actionOne][actionTwo];
	return reward, reward;
end



return ro;