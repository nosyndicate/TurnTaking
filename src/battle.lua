local battle = {};

battle.matrixOne = {{1,0},{0,2}};
battle.matrixTwo = {{2,0},{0,1}};

function battle:playGame(actionOne, actionTwo)
	return self.matrixOne[actionOne][actionTwo], self.matrixTwo[actionOne][actionTwo];
end

return battle;