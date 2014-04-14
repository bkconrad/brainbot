local NeuralNetwork = require('nn')
local Strategy = { verbosity = 0, report = false }

local function report(name, data)
	if Strategy.report then
		local result = name .. ":\n"
		for k,v in pairs(data) do
			result = result .. "- " .. tostring(k) .. ": " .. tostring(v) .. "\n"
		end
		writeToFile('reporting', result, true)
	end
end

local function v(...)
	if Strategy.verbosity >= 1 then
		logprint(...)
	end
end

local function vv(...)
	if Strategy.verbosity >= 2 then
		logprint(...)
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

local UNCERTAINTY_LEARNING_RATE = 0.001
local HIDDEN_LAYERS = 1.0
local PLAN_STATES   = 10
local RECORD_STATES = 3
-- local REWARD_DISCOUNT  = .5^(1/PLAN_STATES) -- Halflife of PLAN_STATES
local REWARD_DISCOUNT = 0.8
local LEARNING_DECAY   = .5^(1/500000)

function Strategy.create(name, numObservations, actions)
	local result = copy(Strategy)
	result.networks = { }
	result.uncertaintyNetworks = { }

	-- Load or create neural networks
	for i,action in ipairs(actions) do
		local knowledgeData = readFromFile(name..'-'..action.name..'.knowledge')
		local uncertaintyData = readFromFile(name..'-'..action.name..'.uncertainty')
		if knowledgeData ~= '' and uncertaintyData ~= '' then
			v('Loading '..name)
			table.insert(result.networks, NeuralNetwork.load(knowledgeData))
			table.insert(result.uncertaintyNetworks, NeuralNetwork.load(uncertaintyData))
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
			table.insert(result.networks, NeuralNetwork.create(numObservations, 1, HIDDEN_LAYERS, (numObservations + 1), 1.0))

			-- The uncertainty networks. There is again one per action, however
			-- these networks are trained in a different manner than the Q value
			-- networks. Uncertainty networks are initialized to 1.0,
			-- representing complete uncertainty regarding their effects. As
			-- each state is assessed, the uncertainty network for the selected
			-- action is trained towards zero. This has the effect of reducing
			-- the uncertainty regarding the action's value in the given state.
			-- 
			-- Some record of state-action pairs is commonly used in Q learning
			-- to intelligently decide which action to take when experiment.
			-- Once we are fairly certain of an action's value in the current
			-- state, we will no longer experiment with it. In the common
			-- (discreet) implementation of Q learning, this is implemented as a
			-- lookup of table mapping state-actions to counts. Because of the
			-- continuous state space, we choose to use a neural network in this
			-- case as well.
			table.insert(result.uncertaintyNetworks, NeuralNetwork.create(numObservations, 1, HIDDEN_LAYERS, (numObservations + 1), UNCERTAINTY_LEARNING_RATE))
			result.uncertaintyNetworks[#result.uncertaintyNetworks]:setWeights(0.5)
		end
	end

	result.name = name
	result.actions = actions
	result.history = { } -- {actionIndex = bestActionIndex, actionConfidence = bestActionConfidence, startingInputs = observations }

	return result
end

function Strategy:plan(observations, experimentation)

	if experimentation == nil then
		experimentation = 1.0
	end

	-- Carefully plan our next move by comparing the expected value of each
	-- action in the current state
	vv(self.name..' planning:')

	-- Pick the best action according to our policy and experimentation settings
	local bestActionIndex = 1
	local bestActionValue = 0
	local bestActionValueWithBonus = 0
	local actionReport = { }
	for i,action in ipairs(self.actions) do

		-- Get the expected value of taking this action in the given state
		local value = self.networks[i]:forewardPropagate(observations)[1]
		actionReport[action.name] = value

		-- Add a bonus to actions with high uncertainty regarding their effects
		local valueWithBonus = value
		if experimentation > 0 then
			valueWithBonus = valueWithBonus + experimentation * self.uncertaintyNetworks[i]:forewardPropagate(observations)[1]
		end

		-- Keep the best action
		if valueWithBonus > bestActionValueWithBonus then
			bestActionIndex = i
			bestActionValue = value
			bestActionValueWithBonus = valueWithBonus
		end
	end

	local phase = {
		startingInputs   = observations,    -- State when this action was chosen
		actionIndex      = bestActionIndex, -- The chosen action
		actionValue      = bestActionValue, -- The expected value of this action
		reward           = 0
	}

	report(self.name, actionReport)

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

	v(self.name..' learning with reward = '..reward)

	reward = math.min(math.max(reward, 0), 1)

	-- Evaluate old plans
	local lastReward = reward
	local lastValue = 0
	for i = #self.history,1,-1 do

		local phase = self.history[i]
		local thisReward = phase.reward + REWARD_DISCOUNT * lastReward

		-- Reinforced reward
		local desiredOutputs = { thisReward + lastValue - phase.actionValue }
		self.networks[phase.actionIndex].learningRate = self.networks[phase.actionIndex].learningRate * LEARNING_DECAY
		self.networks[phase.actionIndex]:backwardPropagate(phase.startingInputs, desiredOutputs)
		self.uncertaintyNetworks[phase.actionIndex]:backwardPropagate(phase.startingInputs, { math.abs(desiredOutputs[1] - phase.actionValue) })

		lastReward = thisReward
		lastValue = phase.actionValue
	end

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
			writeToFile(self.name..'-'..action.name..'.uncertainty', self.uncertaintyNetworks[i]:save())
		end
	end
end

return Strategy