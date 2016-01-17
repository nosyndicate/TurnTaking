require 'rl'

battle = require 'battle'

local function playGame(playerOne, playerTwo, times)
    totalOne = 0
    totalTwo = 0
  for i = 1,times do
    --print("current iteration = " .. i);
    local actionOne = playerOne:getAction(torch.Tensor{1});
    local actionTwo = playerTwo:getAction(torch.Tensor{1});

    --print("Action One = " .. actionOne .. " Action Two = " .. actionTwo);

    local rewardOne, rewardTwo = battle:playGame(actionOne, actionTwo);

    totalOne = totalOne + rewardOne
    totalTwo = totalTwo + rewardTwo
    --print("rewardOne = ".. rewardOne .. " rewardTwo = " .. rewardTwo);

    playerOne:learn(torch.Tensor{1}, rewardOne);
    playerTwo:learn(torch.Tensor{1}, rewardTwo);

    playerOne:step(torch.Tensor{1}, rewardOne);
    playerOne:step(torch.Tensor{1}, rewardTwo);

end
print("Total 1: " .. totalOne .. " Total 2: " .. totalTwo)
end


local function main()
	local playerOne = rl.Wolf(nil, 1, 2);
	local playerTwo = rl.Wolf(nil, 1, 2);
	playGame(playerOne, playerTwo, 500000);
end



main();
