require 'torch'

-- create a global table for learner package
rl = {};

torch.include('rl','Learner.lua');
torch.include('rl','PolicySearch.lua');
torch.include('rl','Reinforce.lua');
torch.include('rl','DiscretePolicy.lua');






