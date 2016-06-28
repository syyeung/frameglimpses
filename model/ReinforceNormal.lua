local ReinforceNormal, parent = torch.class("nn.ReinforceNormal", "nn.Module")

function ReinforceNormal:__init(stdev)
   parent.__init(self)
   self.stdev = stdev
end

function ReinforceNormal:updateOutput(input)
   self.output:resizeAs(input)
   if self.train ~= false then
      self.output:normal(0, self.stdev)
      -- re-center the means to the input
      self.output:add(input)
   else
      -- use maximum a posteriori (MAP) estimate
      self.output:copy(input)
   end
   return self.output
end

function ReinforceNormal:updateGradInput(input, gradOutput)
   -- f : normal probability density function
   -- x : the sampled values (self.output)
   -- u : mean (mu) (input)
   -- s : standard deviation (sigma) (self.stdev)
   -- derivative of log normal w.r.t. mean
   -- d ln(f(x,u,s))   (x - u)
   -- -------------- = -------
   --      d u           s^2
   self.gradInput:resizeAs(input)
   self.gradInput:copy(self.output):add(-1, input)
   self.gradInput:div(self.stdev^2)
   self.gradInput:cmul(gradOutput)
   return self.gradInput
end
