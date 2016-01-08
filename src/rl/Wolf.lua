require 'torch'


local Wolf, parent = torch.class('rl.Wolf','rl.Learner');

-- interesting thing, WOLF assumes same number of actions in each state I think.
-- assume torch.seed() has been called.
function Wolf:__init(model, numStates, numActions)
	parent.__init(self, model);
	self.deltaWin = .3;
	self.delaLoose = .5;
	self.policy = torch.Tensor(numStates, numActions):fill(1.0/numActions);
	self.avgPolicy = torch.Tensor(numStates, numActions):fill(0.0);
	self.Q = torch.Tensor(numStates, numActions):fill(0.0);
	self.C = torch.Tensor(numStates):fill(0.0);
	print(self.policy);
end

-- receive the state as tensor, return the action
-- the return type is a index of the action (discrete case)
-- or table of continuous values
function Wolf:getAction(s)
	-- From state S select action a with probability policy(s,a) with some exploration.
	rnum = torch.uniform();
	self.currentState = s[1];
	print(self.policy);
	for a=1, self.policy:size(s[1]) do
		if (rnum <= self.policy[s[1]][a]) then
			self.chosenAction = a;
			return a;
		end
		rnum = rnum - self.policy[s[1]][a];
	end
end


-- step function, do something (not necessarily learning) after each step
function Wolf:step(s, r)
	-- update misc stuff
	self.C[self.currentState] = self.C[self.currentState] + 1
	a = 0;
	self.avgPolicy[self.currentState]:apply(function(x)
	                     a = a + 1;
	                     return x +
	                     (1.0 / self.C[self.currentState]) *
	                     (self.policy[self.currentState][a] - x)
	                     end);

	self.policy[self.currentState][self.chosenAction] = self.policy[self.currentState][self.chosenAction] + getPolicyUpdate();
end

function Wolf:getPolicyUpdate()
	delta = 0.0;
	if self.policy[self.currentState].dot(self.Q[self.currentState]) >
		self.avgPolicy[self.currentState].dot(self.Q[self.currentState]) then
		delta = self.deltaWin
	else
		delta = self.deltaLoose
	end

	local maxValue = torch.max(self.Q[self.currentState])
	if (maxValue == self.Q[self.currentState][self.chosenAction]) then
		self.policy[self.currentState][self.chosenAction] =
		self.policy[self.currentState][self.chosenAction] + delta;
	else
		self.policy[self.currentState][self.chosenAction] =
		self.policy[self.currentState][self.chosenAction] +
		(-delta / (self.numActions - 1));
	end
	-- normalize it
	sum = self.policy[self.currentState]:sum();
	self.policy[self.currentState]:apply(function(x) return x / sum; end);

end

-- direct update the policy or value function
function Wolf:learn(s, r)
	-- update Q
	self.Q[self.currentState][self.chosenAction] =
		(1 - self.alpha) * self.Q[self.currentState][self.chosenAction] +
		self.alpha * (r + self.gama * torch.max(self.Q[s[1]]));
end

