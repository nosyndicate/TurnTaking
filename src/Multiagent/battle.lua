local battle = {};

battle.matrixOne = {{1,0},{0,2}};
battle.matrixTwo = {{2,0},{0,1}};
battle.numAction = 2; -- how many available action 
battle.numState = 1; -- how many state variable

function battle:playGame(actionOne, actionTwo)
	return self.matrixOne[actionOne][actionTwo], self.matrixTwo[actionOne][actionTwo];
end



return battle;