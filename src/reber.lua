require 'torch'

local reber = {};

reber.chars = 'BTSXPVE';
reber.embedChars = 'TP';

reber.graph = {{{2,6},{'T','P'}},{{2,3},{'S','X'}},{{4,6},{'S','X'}},{{7},{'E'}},{{4,3},{'V','P'}},{{5,6},{'V','T'}}};

function reber:sequenceToWord(sequence)
	
end

function reber:generateReberSequence(minLength)
	while true do
		local seq = {'B'};
		local node = 1;
		while node~=7 do
			local transitions = self.graph[node];
			-- i is the next character sample from the dict
			local index = torch.random(1,#transitions[1]);
			-- append all the possible char to outchars
			table.insert(seq, transitions[2][index]);  -- what char to insert 
			node = transitions[1][index];  -- next node
		end
		if #seq > minLength then
			return seq;
		end
	end
end


function reber:getSingleEmbeddedReberSequence(minLength)
	local seq = self:generateReberSequence(minLength);
	local index = torch.random(1,#self.embedChars);
	table.insert(seq, 1, 'B');
	table.insert(seq, 2, self.embedChars:sub(index,index));
	table.insert(seq, self.embedChars:sub(index,index));
	table.insert(seq, 'E');

	return seq;
end

function reber:getEmbededReberSequences(n, minLength)
	minLength = minLength or 10;
	examples = {};
	for i=1,n do
		table.insert(examples, getSingleEmbeddedReberSequence(minLength));
	end
	return examples;
end



return reber;