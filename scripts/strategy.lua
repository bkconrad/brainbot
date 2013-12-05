local NeuralNetwork = require('nn')
local Strategy = { }

local function copy(t)
	local result = { }
	for k,v in pairs(t) do
		result[k] = v
	end
	return result
end

local LEARNING_RATE = 0.5
local HIDDEN_LAYERS = 2.0
local DISCOUNT_RATE = 0.7
local PLAN_STATES   = 30

function Strategy.create(name, numObservations, actions)
	local result = copy(Strategy)

	local data = readFromFile(name..'.knowledge')

	if data ~= '' then
		logprint('Loading '..name)
		result.network = NeuralNetwork.load(data)
	else
		result.network = NeuralNetwork.create(numObservations, #actions, HIDDEN_LAYERS, (numObservations + #actions) / 2, LEARNING_RATE)
	end
	result.name = name
	result.history = { }
	result.actions = actions

	return result
end

function Strategy:plan(observations)
	local actionConfidenceLevels = self.network:forewardPropagate(unpack(observations))
	local bestActionIndex = nil
	local bestActionConfidence = 0
	for i,action in ipairs(self.actions) do
		-- logprint(self.actions[i].name..': '..tostring(observations[i]))

		if observations[i] > bestActionConfidence then

			bestActionIndex = i
			bestActionConfidence = observations[i]
		end
	end
	-- logprint()

	local phase = {
		actionIndex = bestActionIndex,
		actionConfidence = bestActionConfidence,
		startingInputs = observations
	}

	-- logprint(self.actions[bestActionIndex].name)

	table.insert(self.history, phase)
	if #self.history > PLAN_STATES then
		table.remove(self.history, 1)
	end
end

function Strategy:learn(reinforcement)
	local relevance = 1

	-- Evaluate old plans
	for i = #self.history,1,-1 do

		local phase = self.history[i]

		-- Set up our desired output
		local desiredOutputs = { [phase.actionIndex] = phase.actionConfidence + (reinforcement * relevance) }
		self.network:backwardPropagate(phase.startingInputs, desiredOutputs)

		local relevance = relevance * DISCOUNT_RATE
	end
end

function Strategy:enact()
	-- Enact latest plan
	self.actions[self.history[#self.history].actionIndex].enact()
end

function Strategy:save()
	writeToFile(self.name..'.knowledge', self.network:save())
end

return Strategy