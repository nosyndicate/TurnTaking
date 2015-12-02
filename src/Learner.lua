local Learner = {};

Learner.id = -1;

function Learner:new(o)
	o = o or {}; -- create a new object if it is not provided
	setmetatable(o, self); -- set the prototype
	self.__index = self;
	return o;
end

function Learner:init()
	print("init "..self.id);
end

function Learner:getAction(iteration) -- the learner knows which iteration they are playing
	return 1;
end


function Learner:learning()
	print("start learning");
end

-- don't forget to return the object we defined
return Learner;