local SimpleBattle, parent = torch.class('drq.SimpleBattle','drq.Game');


-- a repeat game of battle of sex

function SimpleBattle:__init(maxLength, fixedLength, iteration)
	parent.__init();
	-- the length of a single game
	self.maxLength = maxLength or 10;
	self.fixedLength = fixedLength or true;
	self.iteration = iteration or torch.random(2, 10);

	self.gameLength = {};
	self.currentState = -1;
	self.processCounter = -1;
	self.currentIteration = -1;


	self.matrix1 = {{2,0},{0,1}};
	self.matrix2 = {{1,0},{0,2}};

	self.terminal = false;
	
	self:setupGame();
	
end


function Game:play(action1, action2)
	local reward = torch.Tensor{0,0};
	if self.currentState == 1 then
		reward = torch.Tensor({self.matrix1[action1][action2], self.matrix2[action1][action2]});
	end

	self.processCounter = self.processCounter + 1;
	
	-- test if we start a new iteration
	if self.gameLength[self.currentIteration] < self.processCounter then
		self.currentIteration = self.currentIteration + 1;
		self.processCounter = 1;
		
		-- if all iterations are finished, then we done the experiments
		if self.currentIteration > #self.gameLength then
			self.terminal = true;
		end
	end
	
	if self.gameLength[self.currentIteration] > self.processCounter then
		self.currentState = 0;
	elseif self.gameLength[self.currentIteration] > self.processCounter then -- last state in one iteration
		self.currentState = 1;
	end
	
	return torch.Tensor{self.currentState}, reward, self.terminal;

end


function Game:setupGame()

	-- generate the length of each iteration of the game
	for i = 1,self.iteration do
		if self.fixedLength then
			table.insert(self.gameLength, self.maxLength);
		else
			table.insert(self.gameLength, torch.random(1, self.maxLength));
		end
	end
end

function Game:newGame()

	self.processCounter = 1;
	self.currentIteration = 1;
	
	if self.gameLength[self.currentIteration] > self.processCounter then
		self.currentState = 0;
	elseif self.gameLength[self.currentIteration] > self.processCounter then -- last state in one iteration
		self.currentState = 1;
	end
	
	return torch.Tensor{self.currentState},torch.Tensor{0,0},self.terminal;

end
