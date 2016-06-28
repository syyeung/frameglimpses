require 'model.ReinforceNormal'
require 'model.ReinforceCategorical'

local GlimpseAgent = {}
function GlimpseAgent.create_network(input_size, loc_size, loc_embed_size, input_embed_size, rnn_size, num_layers, dropout, loc_std)

  -- there will be 2*n+2 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  table.insert(inputs, nn.Identity()()) -- loc
  for L = 1,num_layers do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local glimpse = inputs[1]
  local loc = inputs[2]

  if dropout > 0 then glimpse = nn.Dropout(dropout)(glimpse) end
  local loc_embed = nn.Linear(loc_size, loc_embed_size)(loc)
  loc_embed = nn.Sigmoid()(loc_embed)

  local lstm_in = nn.JoinTable(2)({glimpse, loc_embed})
  local lstm_in_embed = nn.Linear(input_size + loc_embed_size, input_embed_size)(lstm_in)

  local x, input_size_L
  local outputs = {}
  for L = 1,num_layers do
    -- c,h from previos timesteps
    local prev_c = inputs[L*2+1]
    local prev_h = inputs[L*2+2]
    -- the input to this layer
    if L == 1 then
      x = lstm_in_embed
      input_size_L = input_embed_size
    else
      x = outputs[(L-1)*2]
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end

    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x)
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h)
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)
    -- perform the LSTM update
    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate, in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end

  local next_loc_mean = nn.Linear(rnn_size, 1)(top_h)
  local next_loc = nn.ReinforceNormal(loc_std)(next_loc_mean)

  local output_pred_dist = nn.Linear(rnn_size, 2)(top_h)
  output_pred_dist = nn.SoftMax()(output_pred_dist)
  local output_pred = nn.ReinforceCategorical()(output_pred_dist)

  local pred = nn.Linear(rnn_size, 2)(top_h)
  local conf = nn.Sigmoid()(nn.Linear(rnn_size, 1)(top_h))
  local baseline = nn.Linear(rnn_size, 1)(top_h)

  table.insert(outputs, next_loc)
  table.insert(outputs, output_pred)
  table.insert(outputs, pred)
  table.insert(outputs, conf)
  table.insert(outputs, baseline)

  nngraph.annotateNodes()

  return nn.gModule(inputs, outputs)
end

return GlimpseAgent
