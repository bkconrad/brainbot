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

local HIDDEN_LAYERS = 1.0
local PLAN_STATES   = 30
local RECORD_STATES = 3
local REWARD_DISCOUNT  = .5^(1/PLAN_STATES) -- Halflife of PLAN_STATES
local EXPERIMENT_DECAY = .5^(1/5000)        -- Halflife of 1,000 turns
local LEARNING_DECAY   = .5^(1/5000)

function Strategy.create(name, numObservations, actions)
	local result = copy(Strategy)
	result.networks = { }

	-- Load or create neural networks
	for i,action in ipairs(actions) do
		local data = readFromFile(name..'-'..action.name..'.knowledge')
		if data ~= '' then
			v('Loading '..name)
			table.insert(result.networks, NeuralNetwork.load(data))
		else

			-- The Q Value networks. There is one network per action. These are
			-- trained to give the expected value of each action given a
			-- starting state. Because our state space is continuous rather than
			-- discrete, the Q values can not be stored in a table. Neural
			-- networks give us not only the ability to assess Q values in the
			-- continuous state space, but also to generalize Q values across
			-- similar states.
			--
			-- The output of each network is a single scalar output representing
			-- the expected Q value. When the strategy is assessed, each phase
			-- in action history is visited in reverse order. The network
			-- corresponding to the action in that phase is trained towards the
			-- actual reward received by the latest phase (which lead to an
			-- absorbing state such as dying or killing). The learning rate for
			-- the training of each phase is equal to LEARNING_RATE^i where i is
			-- the (positive) index of each phase relative to the most recent.
			table.insert(result.networks, NeuralNetwork.create(numObservations, 1, HIDDEN_LAYERS, (numObservations + #actions + 1) / 2, 1.0))
		end
	end


	result.name = name
	result.history = { } -- {actionIndex = bestActionIndex, actionConfidence = bestActionConfidence, startingInputs = observations }
	result.actions = actions
	result.experimentationFactor = .5 -- Amount of experimentation to do

	return result
end

function Strategy:plan(observations, allowExperimentation)

	local bestActionIndex = 1
	local bestActionValue = 0

	if allowExperimentation == nil then
		allowExperimentation = true
	end

	if allowExperimentation and math.random() < self.experimentationFactor then
		-- Randomly select an action to take -- this is essential to the
		-- learning process because it eliminates the possibility that there is
		-- something better we could be doing
		bestActionIndex = math.random(1, #self.actions)

		-- Expected value of taking this action in the given state
		bestActionValue = self.networks[bestActionIndex]:forewardPropagate(observations)[1]
	else
		-- Carefully plan our next move by comparing the expected value of each
		-- action in the current state
		vv(self.name..' planning:')

		for i,action in ipairs(self.actions) do

			-- Expected value of taking this action in the given state
			local value = self.networks[i]:forewardPropagate(observations)[1]
			vv(self.actions[i].name..': '..tostring(value))

			-- Keep the best action
			if value > bestActionValue then
				bestActionIndex = i
				bestActionValue = value
			end
		end
	end

	local phase = {
		startingInputs   = observations,    -- State when this action was chosen
		actionIndex      = bestActionIndex, -- The chosen action
		actionValue      = bestActionValue, -- The expected value of this action
		reward           = 0
	}

	vv(self.actions[bestActionIndex].name)
	vv()

	table.insert(self.history, phase)
	if #self.history > PLAN_STATES then
		table.remove(self.history, 1)
	end
end

function Strategy:assess(reward)
	if #self.history > 0 then
		self.history[#self.history].reward = reward
	end
end

function Strategy:learn(reward)
	local relevance = 1
	local learningRate = 1.0

	vv(self.name..' learning with reward = '..reward)

	-- Evaluate old plans
	local lastReward = reward
	local lastValue = 0
	for i = #self.history,1,-1 do

		local phase = self.history[i]

		-- Reinforced reward
		local desiredOutputs = { lastReward + REWARD_DISCOUNT * lastValue - phase.actionValue }
		self.networks[phase.actionIndex].learningRate = self.networks[phase.actionIndex].learningRate * LEARNING_DECAY
		self.networks[phase.actionIndex]:backwardPropagate(phase.startingInputs, desiredOutputs)

		-- Store values for next iteration
		lastReward = phase.reward
		lastValue = phase.actionValue
	end

	self.experimentationFactor = self.experimentationFactor * EXPERIMENT_DECAY

	self:save()

	self.history = { }
end

function Strategy:enact()
	-- Enact latest plan
	if #self.history > 0 then
		self.actions[self.history[#self.history].actionIndex].enact()
	end
end

local gNextSave = 0
function Strategy:save()
	if getMachineTime() > gNextSave then
		gNextSave = getMachineTime() + 10000
		for i,action in ipairs(self.actions) do
			writeToFile(self.name..'-'..action.name..'.knowledge', self.networks[i]:save())
		end
	end
end

return Strategy