local DetLossCriterion, parent = torch.class("nn.DetLossCriterion", "nn.Criterion")

function DetLossCriterion:__init(loc_weight)
   parent.__init(self)
   self.loc_weight = loc_weight or 1
   self.gradInput = {}

   self.gt_locs = torch.Tensor()
   self.loc_diffs = torch.Tensor()
   self.clf_losses = torch.Tensor()
   self.loc_losses = torch.Tensor()
end

function DetLossCriterion:updateOutput(input, target)
   local batch_size = #target
   local input_preds = input[1]
   local used_frames = input[2]
   local seq_len = input[3]
   local num_glimpses = #used_frames

   self.gt_locs = torch.Tensor(batch_size, num_glimpses, 2):zero()
   self.loc_diffs = torch.Tensor(batch_size, num_glimpses, 2):zero()
   self.clf_losses = torch.Tensor(batch_size, num_glimpses, 1):zero()
   self.loc_losses = torch.Tensor(batch_size, num_glimpses, 2):zero()

   for b=1,batch_size do
      local gts = target[b]
      local gt_mapping = torch.Tensor(seq_len):zero()
      local num_gts = #gts
      for i=1,num_gts do
         local cur_gt_start = gts[i][1]
         local mapping_start = 0
         if i>1 then
            local prev_gt_end = gts[i-1][2]
            mapping_start = math.floor((prev_gt_end + cur_gt_start)/2)
         end
         local cur_gt_end = gts[i][2]
         local mapping_end = seq_len
         if i < num_gts then
            local next_gt_start = gts[i+1][1]
            mapping_end = math.floor((next_gt_start + cur_gt_end)/2)
         end
         for s=mapping_start+1,mapping_end do
            gt_mapping[s] = i
         end
      end

      local pred = torch.Tensor(num_glimpses, 2)
      local conf = torch.Tensor(num_glimpses, 1)
      for s=1,num_glimpses do
         pred[s]:copy(input_preds[s][2][b])
         conf[s]:copy(input_preds[s][3][b])
      end

      if num_gts > 0 then
         self.clf_losses[b] = torch.log(conf:add(1e-12))
      else
         self.clf_losses[b] = torch.log(conf:mul(-1):add(1):add(1e-12))
      end

      if num_gts > 0 then
         for s=1,num_glimpses do
            local gt_idx = gt_mapping[used_frames[s][b][1]]
            self.gt_locs[b][s][1] = gts[gt_idx][1] / seq_len
            self.gt_locs[b][s][2] = gts[gt_idx][2] / seq_len
         end
         self.loc_diffs[b] = pred - self.gt_locs[b]
         self.loc_losses[b] = torch.pow(self.loc_diffs[b],2):mul(self.loc_weight)
      end
   end
   self.clf_losses:mul(-1) -- neg log likelihood
   self.output = self.clf_losses + torch.sum(self.loc_losses,3)
   return self.output
end

function DetLossCriterion:updateGradInput(input, target)
   local seq_len = #input
   local batch_size = input[1][1]:size(1)

   self.gradInput = {}
   for s=1,seq_len do
      self.gradInput[s] = {}
      self.gradInput[s][1] = torch.Tensor(batch_size,2):zero()
      self.gradInput[s][2] = torch.Tensor(batch_size,1):zero()

      local conf = input[s][3]
      for b=1,batch_size do
         local gts = target[b]
         local num_gts = #gts
         if num_gts > 0 then
            self.gradInput[s][1][b] = self.loc_diffs[b][s]:mul(2):mul(self.loc_weight)
            self.gradInput[s][2][b] = -1 / (conf[b][1] + 1e-12)
         else
            self.gradInput[s][1][b] = 0
            self.gradInput[s][2][b] = 1 / (1 - conf[b][1] + 1e-12)
         end
      end
   end

   return self.gradInput
end
