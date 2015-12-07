require 'torch'

local reber = {};

reber.chars = 'BTSXPVE';
reber.tensorDict = torch.eye(7);
reber.charToNum = {B = 1, T = 2, S = 3, X = 4, P = 5, V = 6, E = 7};
reber.embedChars = 'TP';


reber.graph = {{{2,6},{'T','P'}},{{2,3},{'S','X'}},{{4,6},{'S','X'}},{{7},{'E'}},{{4,3},{'V','P'}},{{5,6},{'V','T'}}};

function reber:sequenceToWord(sequence)
	local string = ""
	for i = 1,#sequence do
		string = string..sequence[i];
	end
	return string;
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

function reber:getEmbeddedReberSequences(n, minLength)
	minLength = minLength or 10;
	examples = {};
	for i= 1, n do
		table.insert(examples, getSingleEmbeddedReberSequence(minLength));
	end
	return examples;
end

-- convert an sequence into a two-d array, each column is one letter, since there are 7 letters, thus there any 7 rows totally
function reber:sequenceToTensor(sequence)  
	local tensorSeq = self.tensorDict:index(2,torch.LongTensor{1}); -- get the first column
	for i = 2, #sequence do
		local char = sequence[i];
		local index = self.charToNum[char];
		local temp = self.tensorDict:index(2,torch.LongTensor{index})  -- get the column from the dict
		tensorSeq = torch.cat(tensorSeq, temp, 2); -- put the column vectors from left to right
	end
	return tensorSeq;
end

function reber:sequenceToOutputClass(sequence)
	local classSeq = {};
	for i = 2,#sequence do
		classSeq[#classSeq+1] = self.charToNum[sequence[i]];
	end
	return classSeq;
end

function reber:getSingleTrainingTuple(minLength)
	local sequence = self:getSingleEmbeddedReberSequence(minLength);
	local length = #sequence;
	local tensorSequence = self:sequenceToTensor(sequence);
	local input = tensorSequence[{{},{1,length-1}}];-- get the first and second to the last sequence
	local output = self:sequenceToOutputClass(sequence);
	return input, output, length;
end

function reber:getTrainingSamples(num, minLength)
	local inputSet = {};
	local outputSet = {};
	local maxLength = -1; -- length cannot be negative
	for i = 1, num do
		local input,output,l = self:getSingleTrainingTuple(minLength)
		table.insert(inputSet, input);
		table.insert(outputSet, output);
		maxLength = math.max(l, maxLength);
	end
	
	return inputSet, outputSet, maxLength;
end




return reber;