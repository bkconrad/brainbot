local NeuralNetwork = require('nn')
local Strategy = { verbosity = 0 }

local function v(...)
	if Strategy.verbosity >= 1 then
		logprint(unpack(arg))
	end
end

local function vv(...)
	if Strategy.verbosity >= 2 then
		logprint(unpack(arg))
	end
end

local function copy(t)
	local result = { }
	for k,v in pairs(t) do
		result[k] = v
	end
	return result
end

local function cat(t1, t2)
	local result = { }
	for i,v in ipairs(t1) do
		table.insert(result, v)
	end
	for i,v in ipairs(t2) do
		table.insert(result, v)
	end
	return result
end

local LEARNING_RATE = 0.5
local HIDDEN_LAYERS = 1.0
local DISCOUNT_RATE = 0.7
local ASSESSMENT_DISCOUNT = 0.9
local PLAN_STATES   = 30
local RECORD_STATES = 3

function Strategy.create(name, numObservations, actions)
	local result = copy(Strategy)

	local data = readFromFile(name..'.knowledge')

	if data ~= '' then
		v('Loading '..name)
		result.network = NeuralNetwork.load(data)
	else

		-- The Q Value network. A Map of State, Action combinations to expected rewards.
		-- Whenever a reward is received, each phase in the plan is visited in reverse
		-- order. This network is trained to expect the given reward from the given
		-- state action pair, at a learning rate equal to ASSESSMENT_DISCOUNT^i where i
		-- is the index of this phase, starting from the most recently executed phase.
		-- This means older phases have less impact on our assessment training.
		--
		-- This network is used to search for the optimal policy. During training of the
		-- policy network, each recorded state is combined with each possible action,
		-- then run through the assessment network. The State, Action pair with the
		-- highest expected reward by our assessment is trained for.
		result.network = NeuralNetwork.create(numObservations + #actions, 1, HIDDEN_LAYERS, (numObservations + #actions + 1) / 2, LEARNING_RATE)
	end


	result.name = name
	result.history = { } -- {actionIndex = bestActionIndex, actionConfidence = bestActionConfidence, startingInputs = observations }
	result.actions = actions
	result.experimentationFactor = .2 -- Amount of experimentation to do

	return result
end

function Strategy:plan(observations)

	local bestActionIndex = 1
	local bestActionValue = 0
	local chosenActions = { } -- 0.0 for foregone options, 1.0 for the selected one

	for i,v in ipairs(self.actions) do
		-- Initialize all values to 0.0
		chosenActions[i] = 0.0
	end

	if math.random() < self.experimentationFactor then
		-- Randomly try something
		bestActionIndex = math.random(1, #self.actions)
		chosenActions[bestActionIndex] = 1.0
		-- Combined State + Action table
		bestActionValue = (self.network:forewardPropagate(unpack(cat(observations, chosenActions))))[1]
	else
		-- Carefully plan our next move by comparing the expected value of each
		-- action in the current state
		vv(self.name..' planning:')
		for i,action in ipairs(self.actions) do

			-- Set this action as chosen
			chosenActions[i] = 1.0

			-- Get the Q value
			value = (self.network:forewardPropagate(unpack(cat(observations, chosenActions))))[1]
			vv(self.actions[i].name..': '..tostring(value))

			-- Keep the best action
			if value > bestActionValue then
				bestActionIndex = i
				bestActionValue = value
			end

			-- Set this action as not chosen to prepare for the next iteration
			chosenActions[i] = 0.0
		end
	end

	local phase = {
		startingInputs   = observations,    -- State when this action was chosen
		actionIndex      = bestActionIndex, -- The chosen action
		actionValue      = bestActionValue, -- The expected value of this action
	}

	vv(self.actions[bestActionIndex].name)
	vv()

	-- self.experimentationFactor = self.experimentationFactor * 0.8

	table.insert(self.history, phase)
	if #self.history > PLAN_STATES then
		table.remove(self.history, 1)
	end
end

function Strategy:learn(reinforcement)
	local relevance = 1
	local chosenActions = { } -- 0.0 for foregone options, 1.0 for the selected one

	for i,v in ipairs(self.actions) do
		-- Initialize all values to 0.0
		chosenActions[i] = 0.0
	end

	self.network._learningRate = 0.8

	-- Evaluate old plans
	local lastReward = reinforcement
	for i = #self.history,1,-1 do

		local phase = self.history[i]
		chosenActions[phase.actionIndex] = 1.0

		-- Reinforced reward
		local desiredOutputs = { reinforcement }
		self.network:backwardPropagate(cat(phase.startingInputs, chosenActions), desiredOutputs)
		chosenActions[phase.actionIndex] = 0.0
		self.network._learningRate = self.network._learningRate * DISCOUNT_RATE
	end

	self:save()

	-- self.history = { }
end

function Strategy:enact()
	-- Enact latest plan
	self.actions[self.history[#self.history].actionIndex].enact()
end

function Strategy:save()
	writeToFile(self.name..'.knowledge', self.network:save())
end

return Strategy