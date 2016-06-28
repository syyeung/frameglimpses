local MSELossCriterion, parent = torch.class("nn.MSELossCriterion", "nn.Criterion")

function MSELossCriterion:__init(module, criterion)
   parent.__init(self)
   self.diff = torch.Tensor()
end

function MSELossCriterion:updateOutput(input, target)
   self.diff = input - target
   self.output = torch.pow(self.diff, 2)
   return self.output
end

function MSELossCriterion:updateGradInput(inputTable, target)
   self.gradInput = torch.mul(self.diff,2)
   return self.gradInput
end
