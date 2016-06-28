require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'util.DataHandler'
require 'model.DetRewardCriterion'
require 'model.DetLossCriterion'
require 'model.MSELossCriterion'
local rapidjson = require 'rapidjson'
local model_utils = require 'util.model_utils'
local GlimpseAgent = require 'model.GlimpseAgent'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a temporal glimpse agent for end-to-end action detection.')
cmd:text()
cmd:text('Options')

-- data files
cmd:option('-train_data_file','','hdf5 file containing training data for all classes')
cmd:option('-train_meta_file','','json file containing meta data about training video chunks')
cmd:option('-val_data_file','','hdf5 file containing validation data')
cmd:option('-val_meta_file','','json file containing meta data about validation video chunks')
cmd:option('-val_vids_file','','text file containing names of validation video names')
cmd:option('-class_mapping_file','thumos_class_mapping.txt','text file mapping class idx to non-contiguous dataset (e.g. Thumos) class idx')

-- model params
cmd:option('-data_dim', 4096, 'input feature dim')
cmd:option('-loc_size', 1, 'location dim')
cmd:option('-loc_embed_size', 1, 'location embed dim')
cmd:option('-input_embed_size', 1024, 'input embed dim')
cmd:option('-rnn_size', 1024, 'LSTM internal state dim')
cmd:option('-num_layers', 3, 'number of layers in the LSTM')
cmd:option('-loc_std', 0.08, 'std dev of location for reinforce sampling')
cmd:option('-loc_weight', 1, 'loc vs. classification weight in loss')
cmd:option('-reward_weight', 1, 'reward scale')
cmd:option('-num_glimpses',6,'number of glimpses (LSTM timesteps)')
cmd:option('-num_classes',20,'number of classes in the data files (used to select pos/neg examples for each batch)')
cmd:option('-pos_class',1,'class to train a classifier for')
cmd:option('-seq_len',50,'number of frames in each video chunk')
cmd:option('-batch_size_pos',50,'number of examples from pos class in each batch')
cmd:option('-batch_size_neg',2,'number of examples from each neg class in each batch')
cmd:option('-fp_reward',-0.1,'false positive reward')
cmd:option('-fn_reward',-10,'false negative reward')

-- optimization
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0.5,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-max_epochs',8,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')

-- bookkeeping
cmd:option('-seed',1,'torch manual random number generator seed')
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every',15,'every how many iterations should we evaluate on validation data?')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')

cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

-- make sure output directory exists
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

local classMapping = {}
if string.len(opt.class_mapping_file) > 0 then
  local f = io.open(opt.class_mapping_file)
  classIter = 1
  for line in f:lines() do
      local words = {}
      for word in line:gmatch("%w+") do table.insert(words, word) end
      classMapping[classIter] = tonumber(words[2])
      classIter = classIter + 1
  end
  f:close()
end

-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 then
    print('initializing gpu...')
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
    print('gpu initialization done.')
end

-----------------------------------------
-----------------------------------------
-- Initialize model
-----------------------------------------
-- -- create the data loader class
dh_args = torch.deserialize(torch.serialize(opt))
local data_handler = DataHandler(dh_args)

-- define the model: prototypes for one timestep, then clone them in time
local do_random_init = true
if string.len(opt.init_from) > 0 then
    print('loading glimpse agent from checkpoint ' .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    protos = checkpoint.protos
    do_random_init = false
else
    print('creating a glimpse agent with ' .. opt.num_layers .. ' layers')
    protos = {}
    protos.rnn = GlimpseAgent.create_network(opt.data_dim, opt.loc_size, opt.loc_embed_size, opt.input_embed_size, opt.rnn_size, opt.num_layers, opt.dropout, opt.loc_std)
    protos.reward_criterion = nn.DetRewardCriterion(opt.reward_weight, opt.fn_reward, opt.fp_reward, opt.no_glimpse_decision, opt.no_output_decision)
    protos.pred_loss_criterion = nn.DetLossCriterion(opt.loc_weight, opt.no_localization)
    protos.baseline_loss_criterion = nn.MSELossCriterion()
end

-- the initial state of the cell/hidden states
init_state = {} --2xnum_layers + 1(loc)
-- location state
local loc_init = torch.zeros(data_handler.batch_size, opt.loc_size)
if opt.gpuid >= 0 then
    loc_init = loc_init:cuda()
end
table.insert(init_state, loc_init:clone())

local init_hidden_offset = 1
for L=1,opt.num_layers do
    local h_init = torch.zeros(data_handler.batch_size, opt.rnn_size)
    if opt.gpuid >=0 then
        h_init = h_init:cuda()
    end
    -- cell state
    table.insert(init_state, h_init:clone())
    -- hidden state
    table.insert(init_state, h_init:clone())
end

-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
end

-- put the above things into one flattened parameters tensor
--params, grad_params = model_utils.combine_all_parameters(protos.rnn)
params, grad_params = protos.rnn:getParameters()

-- initialization
if do_random_init then
params:uniform(-0.07, 0.07) -- small numbers uniform
end

print('number of parameters in the model: ' .. params:nElement())
-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name,proto in pairs(protos) do
    if name == 'rnn' then
        print('cloning ' .. name)
        clones[name] = model_utils.clone_many_times(proto, opt.num_glimpses, not proto.parameters)
    end
end
local init_state_global = model_utils.clone_list(init_state)


-----------------------------------------
-----------------------------------------
-- Forward and backward pass
-----------------------------------------
function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    local x, y, cont = data_handler:next_train_batch()
    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:cuda()
        cont = cont:cuda()
    end
    ------------------- forward pass -------------------
    local rnn_state = {[0] = init_state_global}
    local predictions = {}           -- softmax outputs
    local used_frames = {}
    local input_data = torch.Tensor(data_handler.batch_size, opt.num_glimpses, opt.data_dim)
    if opt.gpuid >= 0 then
        input_data = torch.CudaTensor(data_handler.batch_size, opt.num_glimpses, opt.data_dim)
    end
    for t=1,opt.num_glimpses do
        clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        local cur_loc = rnn_state[t-1][1]
        local cur_frame = torch.clamp(torch.round(cur_loc*opt.seq_len),1,opt.seq_len)
        table.insert(used_frames, cur_frame)
        for b=1,data_handler.batch_size do
            input_data[b][t]:copy(x[b][cur_frame[b][1]])
        end
        local lst = clones.rnn[t]:forward{input_data[{{}, t}], unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        local next_loc = lst[#lst-4]
        local use_pred = lst[#lst-3]
        local pred = lst[#lst-2]
        local conf = lst[#lst-1]
        local baseline = lst[#lst]

        table.insert(rnn_state[t], next_loc)
        for i=1,#init_state-init_hidden_offset do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output

        predictions[t] = {}
        table.insert(predictions[t], use_pred)
        table.insert(predictions[t], pred)
        table.insert(predictions[t], conf)
        table.insert(predictions[t], baseline)
    end

    local reward = protos.reward_criterion:forward(predictions, y)
    local loss = protos.pred_loss_criterion:forward({predictions,used_frames,opt.seq_len}, y)
    local baseline_loss = protos.baseline_loss_criterion:forward(predictions[opt.num_glimpses][4], reward:cuda())

    local avg_reward = torch.sum(reward)/data_handler.batch_size
    local avg_loss = torch.sum(loss)/data_handler.batch_size

    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state = {[opt.num_glimpses] = {}}
    for i=init_hidden_offset+1,#init_state do
        table.insert(drnn_state[opt.num_glimpses], init_state[i]:clone():zero())
    end
    local daction = protos.reward_criterion:backward(predictions,y)
    local doutput = protos.pred_loss_criterion:backward(predictions,y)
    local dbaseline = protos.baseline_loss_criterion:backward(predictions[opt.num_glimpses][4], reward)

    for t=opt.num_glimpses,1,-1 do
        local doutput_t = doutput[t]
        local daction_t = daction[t]
        local dbaseline_t
        if t == opt.num_glimpses then
            dbaseline_t = dbaseline
        else
            dbaseline_t = torch.Tensor(data_handler.batch_size, 1):zero()
        end

        if opt.gpuid >= 0 then
            daction_t[1] = daction_t[1]:cuda() -- next_loc
            daction_t[2] = daction_t[2]:cuda() -- use_pred
            doutput_t[1] = doutput_t[1]:cuda() -- pred
            doutput_t[2] = doutput_t[2]:cuda() -- conf
            dbaseline_t = dbaseline_t:cuda() --baseline
        end
        table.insert(drnn_state[t], daction_t[1]) --next_loc
        table.insert(drnn_state[t], daction_t[2]) --use_pred
        table.insert(drnn_state[t], doutput_t[1]) --pred
        table.insert(drnn_state[t], doutput_t[2]) --conf
        table.insert(drnn_state[t], dbaseline_t) -- lstm baseline
        local dlst = clones.rnn[t]:backward({input_data[{{}, t}], unpack(rnn_state[t-1])}, drnn_state[t])
        drnn_state[t-1] = {}
        for k,v in pairs(dlst) do
            if k > (1+init_hidden_offset) then -- k == 1 is gradient on x, which we dont need, k=2 is loc
                drnn_state[t-1][k-2] = v
            end
        end
    end
    -- clip gradient element-wise
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    return {avg_reward, avg_loss}, grad_params
end


-----------------------------------------
-----------------------------------------
-- Evaluate validation split
-----------------------------------------
function eval_val_split(epoch, randomize_ap)

    print(string.format('Evaluating validation split...'))
    local randomize_ap = randomize_ap or false
    data_handler:reset_data_ptr('val', 1)
    local orderedVidNames = {}
    local n = 0

    local f = io.open(opt.val_vids_file)
    for line in f:lines() do
        local vidName = line:gsub("%s+", "")
        table.insert(orderedVidNames, vidName)
    end
    f:close()
    n = data_handler.num_val_batches

    local numOrderedVids = #orderedVidNames
    local all_gts = {}
    local all_dets = {}
    local all_confs = {}
    local all_preds = {}
    for i=1,numOrderedVids do
        local vidName = orderedVidNames[i]
        all_gts[vidName] = {}
        all_dets[vidName] = {}
        all_confs[vidName] = {}
        all_preds[vidName] = {}
    end

    local total_avg_reward = 0
    local total_avg_loss = 0
    for test_iter = 1,n do
        print(string.format('Evaluating val batch %d/%d', test_iter, n))
        local x, y, cont, meta = data_handler:next_val_batch()
        local num_batch_examples = #meta

        local rnn_state = {[0] = init_state_global}
        local predictions = {}
        local used_frames = {}
        local input_data = torch.Tensor(data_handler.batch_size, opt.num_glimpses, opt.data_dim)
        if opt.gpuid >= 0 then
            input_data = torch.CudaTensor(data_handler.batch_size, opt.num_glimpses, opt.data_dim)
        end
        for t=1,opt.num_glimpses do
            clones.rnn[t]:evaluate() -- make sure we are in correct mode (this is cheap, sets flag)
            local cur_loc = rnn_state[t-1][1]
            local cur_frame = torch.clamp(torch.round(cur_loc*opt.seq_len),1,opt.seq_len)
            table.insert(used_frames, cur_frame)
            for b=1,data_handler.batch_size do
                input_data[b][t]:copy(x[b][cur_frame[b][1]])
            end
            local lst = clones.rnn[t]:forward{input_data[{{}, t}], unpack(rnn_state[t-1])}

            rnn_state[t] = {}
            local next_loc = lst[#lst-4]
            local use_pred = lst[#lst-3]
            local pred = lst[#lst-2]
            local conf = lst[#lst-1]
            local baseline = lst[#lst]

            table.insert(rnn_state[t], next_loc)
            for i=1,#init_state-init_hidden_offset do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output

            predictions[t] = {}
            table.insert(predictions[t], use_pred)
            table.insert(predictions[t], pred)
            table.insert(predictions[t], conf)
            table.insert(predictions[t], baseline)
        end

        local reward = protos.reward_criterion:forward(predictions, y)
        local det_scores = protos.reward_criterion.det_scores
        local loss = protos.pred_loss_criterion:forward({predictions,used_frames,opt.seq_len}, y)

        local avg_reward = torch.sum(reward)/data_handler.batch_size
        local avg_loss = torch.sum(loss)/data_handler.batch_size

        total_avg_reward = total_avg_reward + avg_reward
        total_avg_loss = total_avg_loss + avg_loss

        local output_probs = {}
        for t=1,opt.num_glimpses do
            for i,node in ipairs(clones.rnn[t].forwardnodes) do
                if node.data.annotations.name == 'output_pred_dist' then
                    local output_prob = node.data.module.output:narrow(2,2,1)
                    table.insert(output_probs, output_prob)
                end
            end
        end

        for b=1, num_batch_examples do
            local vidName = meta[b]['vidName']
            local vidStartFrame = meta[b]['seq'][1]

            for t=1,opt.num_glimpses do
                local pred = {}
                local use_pred = predictions[t][1][b][2]
                local use_pred_prob = output_probs[t][b][1]
                local det_start = predictions[t][2][b][1]
                local det_end = predictions[t][2][b][2]
                local conf = predictions[t][3][b][1]
                if conf ~= conf then
                    conf = 0
                end

                local frame = used_frames[t][b][1]

                pred['use_pred'] = use_pred
                pred['use_pred_prob'] = use_pred_prob
                pred['det'] = {det_start, det_end}
                pred['conf'] = conf

                pred['det_score'] = det_scores[b][t]
                pred['frame'] = frame
                pred['chunk_start_frame'] = vidStartFrame
                pred['glimpse_idx'] = t
                table.insert(all_preds[vidName], pred)
            end
        end

        for b=1,num_batch_examples do
            local gts = y[b]
            local vidName = meta[b]['vidName']
            local vidStartFrame = meta[b]['seq'][1]
            for i=1,#gts do
                local gt_start = vidStartFrame + math.min(gts[i][1], gts[i][2]) - 1
                local gt_end = vidStartFrame + math.max(gts[i][1], gts[i][2]) - 1
                local cur_vid_gts = all_gts[vidName]
                local merge_idx = 0
                for cp=1,#cur_vid_gts do
                    cp_start = cur_vid_gts[cp][1]
                    cp_end = cur_vid_gts[cp][2]
                    if gt_start <= cp_end and gt_end >= cp_start then
                        merge_idx = cp
                    elseif gt_start <= cp_end + 2 then
                        merge_idx = cp
                    end
                end
                if merge_idx > 0 then
                    local cp_start = cur_vid_gts[merge_idx][1]
                    local cp_end = cur_vid_gts[merge_idx][2]
                    local new_gt_start = math.min(gt_start, cp_start)
                    local new_gt_end = math.max(gt_end, cp_end)
                    all_gts[vidName][merge_idx] = {new_gt_start, new_gt_end}
                else
                    table.insert(all_gts[vidName], {gt_start, gt_end})
                end
            end

            for t=1,opt.num_glimpses do
                local use_det_t = predictions[t][1][b][2]
                local det_t = torch.round(torch.mul(predictions[t][2][b], opt.seq_len))
                local conf_t = predictions[t][3][b][1]

                if use_det_t == 1 then
                    local det_start = vidStartFrame + math.min(det_t[1], det_t[2]) - 1
                    local det_end = vidStartFrame + math.max(det_t[1], det_t[2]) - 1

                    cur_vid_dets = all_dets[vidName]
                    local merge_idx = 0
                    for cp=1,#cur_vid_dets do
                        cp_start = cur_vid_dets[cp][1]
                        cp_end = cur_vid_dets[cp][2]
                        if det_start <= cp_end and det_end >= cp_start then
                            merge_idx = cp
                        elseif det_start <= cp_end + 2 then
                            merge_idx = cp
                        end
                    end
                    if merge_idx > 0 then
                        local cp_start = cur_vid_dets[merge_idx][1]
                        local cp_end = cur_vid_dets[merge_idx][2]
                        local new_det_start = math.min(det_start, cp_start)
                        local new_det_end = math.max(det_end, cp_end)
                        local new_conf = math.max(conf_t, all_confs[vidName][merge_idx])
                        all_dets[vidName][merge_idx] = {new_det_start, new_det_end}
                        all_confs[vidName][merge_idx] = new_conf
                    else
                        table.insert(all_dets[vidName], {det_start, det_end})
                        table.insert(all_confs[vidName], conf_t)
                    end
                end
            end
            ::batch_continue::
        end
        ::iter_continue::
    end
    total_avg_reward = total_avg_reward / n
    total_avg_loss = total_avg_loss / n

    -- write detections in THUMOS format
    local total_num_gt = 0
    local tpconf = {}
    local fpconf = {}
    local dets_filename = string.format('%s/%02d_val_detections_epoch%.2f.txt', opt.checkpoint_dir, opt.pos_class, epoch)
    local dets_file = io.open(dets_filename, "w")
    for i=1,numOrderedVids do
        local vidName = orderedVidNames[i]
        gts = all_gts[vidName]
        dets = all_dets[vidName]
        confs = all_confs[vidName]

        for di = 1,#dets do
            local det_start = dets[di][1]/5 -- at 5 fps
            local det_end = dets[di][2]/5
            conf = confs[di]
            local mappedClassIdx = opt.pos_class
            if string.len(opt.class_mapping_file) > 0 then
              mappedClassIdx = classMapping[opt.pos_class]
            end
            dets_file:write(string.format('%s\t%.1f\t%.1f\t%d\t%.2f\n', vidName, det_start, det_end, mappedClassIdx, conf))
        end

        total_num_gt = total_num_gt + #gts
        if #dets > 0 then
            local indfree = torch.ones(#dets)
            local ov = interval_overlap(gts, dets)

            for k=1,#gts do
                local indfree_idxs = torch.nonzero(indfree)
                if indfree_idxs:dim() == 0 then -- there are no free indices
                   goto continue_gt
                end
                local free_dets = indfree_idxs:select(2,1)
                local free_ov = ov[k]:index(1,free_dets)
                local max_ov, max_idx = torch.max(free_ov,1)

                local free_dets = torch.nonzero(indfree):select(2,1)
                local free_ov = ov[k]:index(1,free_dets)
                local max_ov, max_idx = torch.max(free_ov,1)
                if max_ov[1] > 0.5 then
                   max_idx = free_dets[max_idx[1]]
                   indfree[max_idx] = 0
                end
                ::continue_gt::
            end

            for i=1,#dets do
                if confs[i] ~= confs[i] then -- nan
                    confs[i] = 0
                end
                if indfree[i] == 0 then --is tp
                    table.insert(tpconf, confs[i])
                else -- fp
                    table.insert(fpconf, confs[i])
                end
            end
        end

    end
    dets_file:close()

    -- compute ap
    local ap = 0
    if randomize_ap then
        local num_random_perms = 10
        for i = 1,num_random_perms do
            print(string.format('randomized map computation %d', i))
            local rand_ap = compute_ap(tpconf, fpconf, total_num_gt, true)
            ap = ap + rand_ap
        end
        ap = ap / num_random_perms
    else
        ap = compute_ap(tpconf, fpconf, total_num_gt, false)
    end
    return ap, total_avg_reward, total_avg_loss
end

function compute_ap(tpconf, fpconf, total_num_gt, randomize_ap)
    local num_tp = #tpconf
    local num_fp = #fpconf
    local conf = torch.Tensor(2, num_tp+num_fp):zero()
    for i=1,num_tp do
        conf[1][i] = round(tpconf[i],2)
        conf[2][i] = 1
    end
    for i=1,num_fp do
        conf[1][i+num_tp] = round(fpconf[i],2)
        conf[2][i+num_tp] = 2
    end
    if num_tp+num_fp == 0 then
        return 0
    end
    local _,sorted_idxs=torch.sort(conf[1], true)
    sorted_conf = torch.Tensor(2, num_tp+num_fp)
    for i=1,sorted_idxs:size(1) do
        sorted_conf[1][i] = conf[1][sorted_idxs[i]]
        sorted_conf[2][i] = conf[2][sorted_idxs[i]]
    end
    local sorted_conf_counts = {}
    local sorted_conf_vals = {}
    local prev_conf_val = 0
    for i=1,sorted_idxs:size(1) do
        local conf_val = sorted_conf[1][i]
        local conf_type = sorted_conf[2][i]
        local new_val = false
        if conf_val ~= prev_conf_val then
            new_val = true
        end
        if new_val then
            table.insert(sorted_conf_vals, conf_val)
            sorted_conf_counts[conf_val] = {0, 0}
            prev_conf_val = conf_val
        end
        sorted_conf_counts[conf_val][conf_type] = sorted_conf_counts[conf_val][conf_type] + 1
    end

    local conf_iter = 1
    for i=1,#sorted_conf_vals do
        local conf_val = sorted_conf_vals[i]
        local conf_counts = sorted_conf_counts[conf_val]
        local num_tp_counts = conf_counts[1]
        local num_fp_counts = conf_counts[2]
        local total_conf_counts = num_tp_counts + num_fp_counts

        sorted_conf[{1, {conf_iter, conf_iter+total_conf_counts-1}}]:fill(conf_val)

        if num_tp_counts > 0 then
            sorted_conf[{2, {conf_iter, conf_iter+num_tp_counts-1}}]:fill(1)
        end
        if num_fp_counts > 0 then
            sorted_conf[{2, {conf_iter+num_tp_counts, conf_iter+total_conf_counts-1}}]:fill(2)
        end

        if randomize_ap then
            local shuffle = torch.randperm(total_conf_counts):type('torch.LongTensor')
            sorted_conf[{2, {conf_iter, conf_iter+total_conf_counts-1}}] = sorted_conf[{2, {conf_iter, conf_iter+total_conf_counts-1}}]:index(1, shuffle)
        end
        conf_iter = conf_iter + total_conf_counts
    end
    tp = torch.cumsum(sorted_conf[2]:eq(1):double())
    fp = torch.cumsum(sorted_conf[2]:eq(2):double())
    tmp = sorted_conf[2]:eq(1):double()
    rec = torch.div(tp, total_num_gt)
    prec = torch.cdiv(tp, tp+fp)
    ap = 0
    for i=1,prec:size(1) do
        if tmp[i] == 1 then
            ap = ap + prec[i]
        end
    end
    ap = ap / total_num_gt
    return ap
end


-----------------------------------------
-----------------------------------------
-- Start optimization here
-----------------------------------------
train_rewards = {}
train_losses = {}
val_rewards = {}
val_losses = {}
val_aps = {}

--local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local optim_state = {learningRate =opt.learning_rate, momentum=0.9}
local iterations = math.floor(opt.max_epochs * data_handler.epoch_batches)
local iterations_per_epoch = data_handler.epoch_batches
print(string.format('optimizing... %d iterations / epoch, max %d iterations for %d epochs total', iterations_per_epoch, iterations, opt.max_epochs))

for i = 1, iterations do
    local epoch = i / data_handler.epoch_batches
    local timer = torch.Timer()
    --local _, optim_out = optim.rmsprop(feval, params, optim_state)
    local _, optim_out = optim.sgd(feval, params, optim_state)

    optim_out = optim_out[1]
    local reward = optim_out[1]
    local loss = optim_out[2]

    local time = timer:time().real
    train_rewards[i] = reward
    train_losses[i] = loss

    -- exponential learning rate decay
    if i % data_handler.epoch_batches == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end

    -- every now and then or on last iteration
    if (i % opt.eval_val_every == 0) or (i == iterations) then
        local val_ap, val_reward, val_loss = eval_val_split(epoch, false)
        val_rewards[i] = val_reward
        val_losses[i] = val_loss
        val_aps[i] = val_ap

        print(string.format("Evaluating validation split: reward = %.2f, loss = %.2f, ap = %.2f",
          val_reward, val_loss, val_ap))

        local savefile = string.format('%s/%02d_ep%.2f_rew%.2f_loss%.2f_ap%.2f.t7',
          opt.checkpoint_dir, opt.pos_class, epoch, val_reward, val_loss, val_ap)
        print('Saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.train_rewards = train_rewards
        checkpoint.train_losses = train_losses
        checkpoint.val_rewards = val_rewards
        checkpoint.val_losses = val_losses
        checkpoint.val_aps = val_aps
        checkpoint.i = i
        checkpoint.epoch = epoch
        torch.save(savefile, checkpoint)
    end

    if i % opt.print_every == 0 then
        print(string.format("Iteration %d/%d (epoch %.3f), reward = %6.8f, loss = %6.8f, time/batch = %.2fs",
          i, iterations, epoch, reward, loss, time))
    end
    if i % 10 == 0 then collectgarbage() end
    -- handle early stopping if things are going really bad
    if reward ~= reward then
        print('loss is NaN, aborting...')
        break -- halt
    end
end
