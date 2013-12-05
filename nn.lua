ACTIVATION_RESPONSE = 1

NeuralNetwork = {
	transfer = function( x) return 1 / (1 + math.exp(-x / ACTIVATION_RESPONSE)) end --This is the Transfer function (in this case a sigmoid)
}

function getn(t)
	return #t
end

function copy(t)
	local result = { }
	for k,v in pairs(t) do
		result[k] = v
	end
	return result
end

function dump(t)
	for k,v in pairs(t) do
		logprint(k,v)
	end
end

function NeuralNetwork.create( _numInputs, _numOutputs, _numHiddenLayers, _neuronsPerLayer, _learningRate)
	_numInputs = _numInputs or 1
	_numOutputs = _numOutputs or 1
	_numHiddenLayers = _numHiddenLayers or math.ceil(_numInputs/2)
	_neuronsPerLayer = _neuronsPerLayer or math.ceil(_numInputs*.66666+_numOutputs)
	_learningRate = _learningRate or .5
	--order goes network[layer][neuron][wieght]
	local network = copy(NeuralNetwork)
	network.learningRate = _learningRate
	network[1] = {}   --Input Layer
	for i = 1,_numInputs do
		network[1][i] = {}
	end
	for i = 2,_numHiddenLayers+2 do --plus 2 represents the output layer (also need to skip input layer)
		network[i] = {}
		local neuronsInLayer = _neuronsPerLayer
		if i == _numHiddenLayers+2 then
			neuronsInLayer = _numOutputs
		end
		for j = 1,neuronsInLayer do
			network[i][j] = {bias = math.random()*2-1}
			local numNeuronInputs = getn(network[i-1])
			for k = 1,numNeuronInputs do
				network[i][j][k] = math.random()*2-1  --return random number between -1 and 1
			end
		end
	end
	return network
end
	
function NeuralNetwork:forewardPropagate(...)
	if #arg ~= #self[1] and type(arg[1]) ~= "table" then
		error("Neural Network received "..#arg.." input[s] (expected "..#self[1].." input[s])",2)
	elseif type(arg[1]) == "table" and getn(arg[1]) ~= getn(self[1]) then
		error("Neural Network received "..getn(arg[1]).." input[s] (expected "..getn(self[1]).." input[s])",2)
	end
	local outputs = {}
	for i = 1,getn(self) do
		for j = 1,getn(self[i]) do
			if i == 1 then
				if type(arg[1]) == "table" then
					self[i][j].result = arg[1][j]
				else
					self[i][j].result = arg[j]
				end
			else
				self[i][j].result = self[i][j].bias
				for k = 1,getn(self[i][j]) do
					self[i][j].result = self[i][j].result + (self[i][j][k]*self[i-1][k].result)
				end
				self[i][j].result = NeuralNetwork.transfer(self[i][j].result)
				if i == getn(self) then
					table.insert(outputs,self[i][j].result)
				end
			end
		end

	end
	return outputs
end

function NeuralNetwork:backwardPropagate(inputs,desiredOutputs)
	--[[
	if getn(inputs) ~= getn(self[1]) then
		error("Neural Network received "..getn(inputs).." input[s] (expected "..getn(self[1]).." input[s])",2)
	elseif getn(desiredOutputs) ~= getn(self[getn(self)]) then
		error("Neural Network received "..getn(desiredOutputs).." desired output[s] (expected "..getn(self[getn(self)]).." desired output[s])",2)
	end
	]]
	self:forewardPropagate(inputs) --update the internal inputs and outputs
	for i = getn(self),2,-1 do --iterate backwards (nothing to calculate for input layer)
		local tempResults = {}
		for j = 1,getn(self[i]) do
			if desiredOutputs[j] ~= nil then
				if i == getn(self) then --special calculations for output layer
					self[i][j].delta = (desiredOutputs[j] - self[i][j].result) * self[i][j].result * (1 - self[i][j].result)
				else
					local weightDelta = 0
					for k = 1,getn(self[i+1]) do
						if desiredOutputs[k] ~= nil then
							weightDelta = weightDelta + self[i+1][k][j]*self[i+1][k].delta
						end
					end
					self[i][j].delta = self[i][j].result * (1 - self[i][j].result) * weightDelta
				end
			end
		end
	end
	for i = 2,getn(self) do
		for j = 1,getn(self[i]) do
			if desiredOutputs[j] ~= nil then
				self[i][j].bias = self[i][j].delta * self.learningRate
				for k = 1,getn(self[i][j]) do
					if desiredOutputs[k] ~= nil then
						self[i][j][k] = self[i][j][k] + self[i][j].delta * self.learningRate * self[i-1][k].result
					end
				end
			end
		end
	end
end

function NeuralNetwork:save()
	--[[
	File specs:
		|INFO| - should be FF BP NN
		|I| - number of inputs
		|O| - number of outputs
		|HL| - number of hidden layers
		|NHL| - number of neurons per hidden layer
		|LR| - learning rate
		|BW| - bias and weight values
	]]--
	local data = "|INFO|FF BP NN|I|"..tostring(getn(self[1])).."|O|"..tostring(getn(self[getn(self)])).."|HL|"..tostring(getn(self)-2).."|NHL|"..tostring(getn(self[2])).."|LR|"..tostring(self.learningRate).."|BW|"
	for i = 2,getn(self) do -- nothing to save for input layer
		for j = 1,getn(self[i]) do
			local neuronData = tostring(self[i][j].bias).."{"
			for k = 1,getn(self[i][j]) do
				neuronData = neuronData..tostring(self[i][j][k])
				neuronData = neuronData..","
			end
			data = data..neuronData.."}"
		end
	end
	data = data.."|END|"
	return data		
end
function NeuralNetwork.load( data)
	local dataPos = string.find(data,"|")+1
	local currentChunk = string.sub( data, dataPos, string.find(data,"|",dataPos)-1)
	local dataPos = string.find(data,"|",dataPos)+1
	local _inputs, _outputs, _hiddenLayers, _neuronsPerLayer, _learningRate
	local biasWeights = {}
	local errorExit = false
	while currentChunk ~= "END" and not errorExit do
		if currentChuck == "INFO" then
			currentChunk = string.sub( data, dataPos, string.find(data,"|",dataPos)-1)
			dataPos = string.find(data,"|",dataPos)+1
			if currentChunk ~= "FF BP NN" then
				errorExit = true
			end
		elseif currentChunk == "I" then
			currentChunk = string.sub( data, dataPos, string.find(data,"|",dataPos)-1)
			dataPos = string.find(data,"|",dataPos)+1
			_inputs = tonumber(currentChunk)
		elseif currentChunk == "O" then
			currentChunk = string.sub( data, dataPos, string.find(data,"|",dataPos)-1)
			dataPos = string.find(data,"|",dataPos)+1
			_outputs = tonumber(currentChunk)
		elseif currentChunk == "HL" then
			currentChunk = string.sub( data, dataPos, string.find(data,"|",dataPos)-1)
			dataPos = string.find(data,"|",dataPos)+1
			_hiddenLayers = tonumber(currentChunk)
		elseif currentChunk == "NHL" then
			currentChunk = string.sub( data, dataPos, string.find(data,"|",dataPos)-1)
			dataPos = string.find(data,"|",dataPos)+1
			_neuronsPerLayer = tonumber(currentChunk)
		elseif currentChunk == "LR" then
			currentChunk = string.sub( data, dataPos, string.find(data,"|",dataPos)-1)
			dataPos = string.find(data,"|",dataPos)+1
			_learningRate = tonumber(currentChunk)
		elseif currentChunk == "BW" then
			currentChunk = string.sub( data, dataPos, string.find(data,"|",dataPos)-1)
			dataPos = string.find(data,"|",dataPos)+1
			local subPos = 1 
			local subChunk
			for i = 1,_hiddenLayers+1 do
				biasWeights[i] = {}
				local neuronsInLayer = _neuronsPerLayer
				if i == _hiddenLayers+1 then
					neuronsInLayer = _outputs
				end
				for j = 1,neuronsInLayer do
					biasWeights[i][j] = {}
					biasWeights[i][j].bias = tonumber(string.sub(currentChunk,subPos,string.find(currentChunk,"{",subPos)-1))
					subPos = string.find(currentChunk,"{",subPos)+1
					subChunk = string.sub( currentChunk, subPos, string.find(currentChunk,",",subPos)-1)
					local maxPos = string.find(currentChunk,"}",subPos)
					while subPos < maxPos do
						table.insert(biasWeights[i][j],tonumber(subChunk))
						subPos = string.find(currentChunk,",",subPos)+1
						if string.find(currentChunk,",",subPos) ~= nil then
							subChunk = string.sub( currentChunk, subPos, string.find(currentChunk,",",subPos)-1)
						end
					end
					subPos = maxPos+1
				end
			end			
		end
		currentChunk = string.sub( data, dataPos, string.find(data,"|",dataPos)-1)
		dataPos = string.find(data,"|",dataPos)+1
	end
	if errorExit then
		error("Failed to load Neural Network:"..currentChunk,2)
	end
	local network = copy(NeuralNetwork)
	network.learningRate = _learningRate
	network[1] = {}   --Input Layer
	for i = 1,_inputs do
		network[1][i] = {}
	end
	for i = 2,_hiddenLayers+2 do --plus 2 represents the output layer (also need to skip input layer)
		network[i] = {}
		local neuronsInLayer = _neuronsPerLayer
		if i == _hiddenLayers+2 then
			neuronsInLayer = _outputs
		end
		for j = 1,neuronsInLayer do
			network[i][j] = {bias = biasWeights[i-1][j].bias}
			local numNeuronInputs = getn(network[i-1])
			for k = 1,numNeuronInputs do
				network[i][j][k] = biasWeights[i-1][j][k]
			end
		end
	end
	return network
end

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

STATE = {
	target = nil,
	targetingRange = 1000,
	firingRange = 500,
	justDied = false,
	justKilled = false
}

INPUTS = {
	{
		name = "health",
		measure = function()
			return bot:getHealth() or 0
		end
	},

	{
		name = "energy",
		measure = function()
			return bot:getEnergy() or 0
		end
	},

	{
		name = "targetHealth",
		measure = function()
			if STATE.target == nil then
				return 0
			end

			return STATE.target:getHealth() or 0
		end
	},

	{
		name = "canSeeTarget",
		measure = function()
			if STATE.target == nil or not STATE.target:getGeom() or not bot:canSeePoint(STATE.target:getGeom())then
				return 0
			end

			return 1
		end
	},

	{
		name = "haveTarget",
		measure = function()
			if (STATE.target == nil) or (STATE.target:getHealth() == nil) or (STATE.target:getHealth() <= 0) then
				return 0
			end

			return 1
		end
	},
}

ACTIONS = {
	{
		name = "target",
		enact = function()
			STATE.target = bot:findClosestEnemy(STATE.targetingRange)
		end
	},

	{
		name = "engage",
		enact = function()
			if bot == nil or STATE.target == nil then
				return
			end

			local solution = bot:getFiringSolution(STATE.target)

			if solution then
				bot:setAngle(solution)
				bot:fireWeapon(Weapon.Phaser)
			end

			if STATE.target and STATE.target:getPos() then
				bot:setThrustToPt(bot:getWaypoint(STATE.target:getPos()))
			end
		end
	},
	
}

PLAN = { }
PLAN_STATES = 30
PLAN_INTERVAL = 100
NEXT_PLAN_TIME = 0

DISCOUNT_RATE = .7

local knowledge = readFromFile('current')
if knowledge ~= '' then
	NN = NeuralNetwork.load(knowledge)
else
NN = NeuralNetwork.create(#INPUTS, #ACTIONS, 1, (#INPUTS + #ACTIONS) / 2, .2)
end

function main()
	subscribe(Event.MsgReceived)
	subscribe(Event.ScoreChanged)
end

function onMsgReceived()
	logprint('debugging!')
end

local lastWrite = 0
local start = getMachineTime()
function onScoreChanged(scoreChange, teamIndex, player) 
	if getMachineTime() > lastWrite then
		lastWrite = getMachineTime()
		writeToFile('current', NN:save())
	end

	if player == bot:getPlayerInfo() and scoreChange > 0 then
		STATE.justKilled = true
	elseif player ~= bot:getPlayerInfo() and scoreChange > 0 then
		STATE.justDied = true
	end
end


local clock = 0
local deaths = 0
local kills = 0
local record = { }
local RECORD_STATES = 100
function onTick(dt)
	clock = clock + dt

	if bot:getHealth() <= 0 then
		return
	end

	if clock > NEXT_PLAN_TIME then
		local currentInputs = getInputs()

		local reinforcement = 0
		if STATE.justKilled then
			-- logprint('killed')
			kills = kills + 1
			table.insert(record, true)
			reinforcement = 1
		elseif STATE.justDied then
			-- logprint('died')
			deaths = deaths + 1
			table.insert(record, false)
			reinforcement = -1
		end

		if #record > RECORD_STATES then
			table.remove(record, 1)
		end

		if reinforcement ~= 0 then
			logprint('kills '..kills)
			logprint('deaths '..deaths)
			local k, d = 0, 0
			for i,v in ipairs(record) do
				if v then
					k = k + 1
				else
					d = d + 1
				end
			end
			writeToFile("record", tostring(k / d).."\n", true)
			logprint(k/d)

			local relevance = 1

			-- Evaluate old plans
			for i = #PLAN,1,-1 do

				local phase = PLAN[i]

				-- Set up our desired output
				local desiredOutputs = { [phase.actionIndex] = phase.actionConfidence + (reinforcement * relevance) }
				NN:backwardPropagate(phase.startingInputs, desiredOutputs)

				local relevance = relevance * DISCOUNT_RATE
			end
		end

		if STATE.justDied then
			-- The old plan doesn't matter now
			PLAN = { }
		end

		STATE.justDied = false
		STATE.justKilled = false

		-- Pick a new plan
		local outputs = NN:forewardPropagate(unpack(currentInputs))
		local bestActionIndex = nil
		local bestActionConfidence = 0
		for i,action in ipairs(ACTIONS) do
			-- logprint(ACTIONS[i].name..': '..tostring(outputs[i]))

			if outputs[i] > bestActionConfidence then

				bestActionIndex = i
				bestActionConfidence = outputs[i]
			end
		end
		-- logprint()

		local phase = {
			actionIndex = bestActionIndex,
			actionConfidence = bestActionConfidence,
			startingInputs = currentInputs
			-- TODO: factor action confidence in to learning rate?
		}

		-- logprint(ACTIONS[bestActionIndex].name)

		table.insert(PLAN, phase)
		if #PLAN > PLAN_STATES then
			table.remove(PLAN, 1)
		end

		NEXT_PLAN_TIME = clock + PLAN_INTERVAL
	end

	-- Enact latest plan
	ACTIONS[PLAN[#PLAN].actionIndex].enact()
end

function getInputs()
	local result = { }

	for i,input in ipairs(INPUTS) do
		local v = input.measure()
		result[input.name] = v
		result[i] = v
	end

	return result
end