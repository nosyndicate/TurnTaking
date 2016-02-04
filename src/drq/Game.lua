local Game = torch.class('drq.Game');


function Game:__init()

end


function Game:play(action1, action2)
	local state, reward, terminal = nil, nil, nil;
	return state, reward, terminal;
end


function Game:newGame()
	local state, reward, terminal = nil, nil, nil;
	return state, reward, terminal;
end