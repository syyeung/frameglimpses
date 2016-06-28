local ReinforceCategorical, parent = torch.class("nn.ReinforceCategorical", "nn.Module")

function ReinforceCategorical:__init()
   parent.__init(self)
   self._index = torch.Tensor()
end

function ReinforceCategorical:updateOutput(input)
   self.output:resizeAs(input)
   if self.train ~= false then
      -- sample from categorical with p = input
      input.multinomial(self._index, input, 1)
   else -- test-time
      _,self._index=torch.max(input,2)
   end
   -- one hot encoding
   self.output:zero()
   self.output:scatter(2, self._index, 1)
   return self.output
end

function ReinforceCategorical:updateGradInput(input, gradOutput)
   -- f : categorical probability mass function
   -- x : the sampled indices (one per sample) (self.output)
   -- p : probability vector (p[1], p[2], ..., p[k])
   -- derivative of log categorical w.r.t. p
   -- d ln(f(x,p))     1/p[i]    if i = x
   -- ------------ =
   --     d p          0         otherwise
   self.gradInput:resizeAs(input):zero()
   self.gradInput:copy(self.output)
   self.gradInput:cdiv(input)
   self.gradInput:cmul(gradOutput)
   return self.gradInput
end
