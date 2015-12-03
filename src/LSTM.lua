-- adapted from: wojciechz/learning_to_execute on github
require 'nn'
require 'nngraph'


local LSTM = {};

-- Creates one timestep of one LSTM
function LSTM.create(size)     -- if size == 1, there is 8 weights in this layer, not including bias
--[[
	for code like this : x = nn.Identity()(),
	the first () create the module,
 	the second call converts the module to a node in the graph and the argument specifies it’s parent in the graph.
--]]

	local x = nn.Identity()();      -- network input at time step t
	local prevC = nn.Identity()();  -- c at time step t-1
	local prevH = nn.Identity()();  -- h at time step t-1

	function inputSum()
		-- transforms input
		local i2h = nn.Linear(size, size)(x)      -- input to hidden
		-- transforms previous timestep's output
		local h2h = nn.Linear(size, size)(prevH)  -- hidden to hidden
		local sum = nn.CAddTable()({i2h, h2h});
		return sum;
	end

	local inputGate = nn.Sigmoid()(inputSum());
	local forgetGate = nn.Sigmoid()(inputSum());
	local outputGate = nn.Sigmoid()(inputSum());
	local inputTransform = nn.Tanh()(inputSum());

	local nextC = nn.CAddTable()({
		nn.CMulTable()({forgetGate, prevC}),
		nn.CMulTable()({inputGate, inputTransform})
	});
	local nextH = nn.CMulTable()({outputGate, nn.Tanh()(nextC)});

	return nn.gModule({x, prevC, prevH}, {nextC, nextH});
end

return LSTM
