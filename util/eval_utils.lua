function interval_overlap(gts, dets)
  local num_gt = #gts
  local num_det = #dets
  local ov = torch.Tensor(num_gt, num_det)
  for i=1,num_gt do
    for j=1,num_det do
      ov[i][j] = interval_overlap_single(gts[i], dets[j])
    end
  end
  return ov
end

function interval_overlap_single(gt, dt)
  local i1 = gt
  local i2 = dt
  -- union
  local bu = {math.min(i1[1], i2[1]), math.max(i1[2], i2[2])}
  local ua = bu[2] - bu[1]
  -- overlap
  local ov = 0
  local bi = {math.max(i1[1], i2[1]), math.min(i1[2], i2[2])}
  local iw = bi[2] - bi[1]
  if iw > 0 then
    ov = iw / ua
  end
  return ov
end

function round(num, idp)
  local mult = 10^(idp or 0)
  return math.floor(num * mult + 0.5) / mult
end
