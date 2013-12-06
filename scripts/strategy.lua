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

local LEARNING_RATE = 0.5
local HIDDEN_LAYERS = 1.0
local DISCOUNT_RATE = 0.7
local PLAN_STATES   = 30
local RECORD_STATES = 3
local EXPERIMENTATION_FACTOR = .5

function Strategy.create(name, numObservations, actions)
	local result = copy(Strategy)

	local data = readFromFile(name..'.knowledge')

	if data ~= '' then
		v('Loading '..name)
		result.network = NeuralNetwork.load(data)
	else
		result.network = NeuralNetwork.create(numObservations, #actions, HIDDEN_LAYERS, (numObservations + #actions) / 2, LEARNING_RATE)
	end
	result.name = name
	result.history = { } -- {actionIndex = bestActionIndex, actionConfidence = bestActionConfidence, startingInputs = observations }
	result.experiments = { }
	result.actions = actions

	return result
end

function Strategy:plan(observations)
	local actionConfidenceLevels = self.network:forewardPropagate(unpack(observations))
	local bestActionIndex = nil
	local bestActionConfidence = 0

	vv(self.name..' planning:')
	for i,action in ipairs(self.actions) do
		vv(self.actions[i].name..': '..tostring(observations[i]))

		if observations[i] > bestActionConfidence then

			bestActionIndex = i
			bestActionConfidence = observations[i]
		end
	end

	local phase = {
		actionIndex = bestActionIndex,
		actionConfidence = bestActionConfidence,
		startingInputs = observations
	}

	vv(self.actions[bestActionIndex].name)
	vv()

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

function Strategy:_index2weight(index)
	if index < 1 then
		return
	end

	local n = 0
	for i=2,#self.network do                  -- For each layer after the input layer
		for j=1,#self.network[i] do           -- For each neuron in the layer before i
			for k=1,#self.network[i][j] do    -- For each neuron in i
				n = n + 1                     -- Count by one

				if n >= index then
					return i, j, k
				end
			end
		end
	end
end

function Strategy:experiment(reinforcement)
	-- Record current experiment performance via reinforcement
	if #self.experiments > 0 then
		table.insert(self.experiments[#self.experiments].record, reinforcement)
	end

	-- If there is no current experiment, or the current experiment is over
	if #self.experiments == 0 or #self.experiments[#self.experiments].record >= RECORD_STATES then

		-- De-modulate old experiment's weight
		local i, j, k = self:_index2weight(#self.experiments)
		if i and j and k then
			self.network[i][j][k] = self.network[i][j][k] - self.experiments[#self.experiments].hypothesis
		end

		-- Look for the next weight to experiment with, taking parity as 1, then -1
		local parity
		for _parity = 1,-1,-2 do
			parity = _parity
			i, j, k = self:_index2weight(#self.experiments + 1)
		end

		-- If there is another weight to experiment with, then start the experiment
		if i and j and k then
			local experiment = { record = { }, hypothesis = parity * EXPERIMENTATION_FACTOR }
			table.insert(self.experiments, experiment)
			-- Modulate the weight according to the experiment
			self.network[i][j][k] = self.network[i][j][k] + self.experiments[#self.experiments].hypothesis
			vv('starting new experiment')
		else
			-- This round of experiments is over!
			-- Pick the best one and apply it permanently
			-- TODO: Select all changes which performed better than some threshold
			local bestExperimentIndex = 0
			local bestExperimentPerformance = -math.huge
			for i,experiment in ipairs(self.experiments) do
				local performance = 0
				for j,score in ipairs(experiment.record) do
					performance = performance + score
				end

				if performance > bestExperimentPerformance then
					bestExperimentIndex = i
					bestExperimentPerformance = performance
				end
			end

			-- Modulate according to the given hypothesis, permanently
			local i, j, k = self:_index2weight(bestExperimentIndex)
			self.network[i][j][k] = self.network[i][j][k] + self.experiments[bestExperimentIndex].hypothesis

			-- Save
			self:save()

			-- Clear experiments
			self.experiments = { }
			v(logprint('starting new round of experiments'))
		end
	end
end

return Strategy