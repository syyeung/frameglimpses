require 'util.eval_utils'

local DetRewardCriterion, parent = torch.class("nn.DetRewardCriterion", "nn.Criterion")

function DetRewardCriterion:__init(scale, fn_reward, fp_reward)
   parent.__init(self)
   self.scale = scale or 1
   self.fn_reward = fn_reward or -10
   self.fp_reward = fp_reward or -0.1

   self.det_scores = torch.Tensor()
   self.gradInput = {}
end

function DetRewardCriterion:updateOutput(input, target)

   local batch_size = #target
   local seq_len = #input

   self.det_scores = torch.Tensor(batch_size, seq_len):zero()
   local gt_scores = torch.Tensor(batch_size):zero()

   for b=1,batch_size do
      local gts = target[b]
      local use_pred = torch.Tensor(seq_len, 2)
      local pred = torch.Tensor(seq_len, 2)
      local conf = torch.Tensor(seq_len, 1)

      for s=1,seq_len do
         use_pred[s]:copy(input[s][1][b])
         pred[s]:copy(input[s][2][b])
         conf[s]:copy(input[s][3][b])
      end

      local use_pred_true = torch.gt(use_pred:select(2,2),0.5)

      local nonzero_use_pred_true = torch.nonzero(use_pred_true)
      if nonzero_use_pred_true:dim() == 0 then
         if #gts > 0 then
            gt_scores[b] = self.fn_reward
         end
         goto continue_batch
      end

      local use_pred_idxs = nonzero_use_pred_true:select(2,1)
      pred = pred:index(1,use_pred_idxs)
      conf = conf:index(1,use_pred_idxs)

      local sorted_conf,sorted_idxs=torch.sort(conf, 1, true)
      local sorted_pred = {}
      for i=1,sorted_idxs:size(1) do
         table.insert(sorted_pred, pred[sorted_idxs[i][1]])
      end

      if #sorted_pred > 0 then

         local indfree=torch.ones(#sorted_pred)
         local ov = interval_overlap(gts, sorted_pred)

         for k=1,#gts do
            local indfree_idxs = torch.nonzero(indfree)
            if indfree_idxs:dim() == 0 then -- there are no free indices
               goto continue_gt
            end

            local free_dets = indfree_idxs:select(2,1)
            local free_ov = ov[k]:index(1,free_dets)
            local max_ov, max_idx = torch.max(free_ov,1)

            if max_ov[1] > 0.5 then
               max_idx = free_dets[max_idx[1]]
               indfree[max_idx] = 0
               local sorted_idx = sorted_idxs[max_idx][1]
               local orig_idx = use_pred_idxs[sorted_idx]
               self.det_scores[b][orig_idx] = 1
            end
            ::continue_gt::
         end

         local unused_dets = torch.nonzero(indfree)
         if unused_dets:dim() > 0 then
            local num_unused_dets = unused_dets:size(1)
            for i=1,num_unused_dets do
               local sorted_idx = sorted_idxs[unused_dets[i][1]][1]
               local orig_idx = use_pred_idxs[sorted_idx]
               self.det_scores[b][orig_idx] = self.fp_reward
            end
         end
      end
      ::continue_batch::
   end

   self.rewards = torch.sum(self.det_scores,2)
   self.rewards:add(gt_scores)
   self.rewards:mul(self.scale)
   self.output = self.rewards
   return self.output
end

function DetRewardCriterion:updateGradInput(input, target)

   local batch_size = input[1][1]:size(1)
   local seq_len = #input
   local baseline = input[seq_len][4]:double() -- baseline reward
   local rewardsRel = self.rewards -- - baseline
   local rewardsGrad = rewardsRel:mul(-1)

   self.gradInput = {}
   for s=1,seq_len do
      self.gradInput[s] = {}
      self.gradInput[s][1] = torch.repeatTensor(rewardsGrad, 1,1) --dLoc
      self.gradInput[s][2] = torch.repeatTensor(rewardsGrad, 1,2) --dAction
   end
   return self.gradInput
end
