local Learner = {};

Learner.id = -1;

function Learner:new(o)
	o = o or {}; -- create a new object if it is not provided
	setmetatable(o, self); -- set the prototype
	self.__index = self;
	o:init();
	
	return o;
end

function Learner:init()

end

function Learner:getAction(iteration) -- the learner knows which iteration they are playing
	return 1;
end


local function feval(params)
	
end

function Learner:learning()
	print("start learning");
end

-- don't forget to return the object we defined
return Learner;