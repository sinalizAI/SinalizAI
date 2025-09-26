class C3(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  cv1 : __torch__.models.common.___torch_mangle_158.Conv
  cv2 : __torch__.models.common.___torch_mangle_161.Conv
  cv3 : __torch__.models.common.___torch_mangle_164.Conv
  m : __torch__.torch.nn.modules.container.___torch_mangle_172.Sequential
  def forward(self: __torch__.models.common.___torch_mangle_173.C3,
    argument_1: __torch__.torch.nn.modules.activation.___torch_mangle_191.SiLU,
    argument_2: Tensor) -> Tensor:
    cv3 = self.cv3
    cv2 = self.cv2
    m = self.m
    cv1 = self.cv1
    _0 = (cv1).forward(argument_1, argument_2, )
    _1 = (m).forward(argument_1, _0, )
    _2 = (cv2).forward(argument_1, argument_2, )
    input = torch.cat([_1, _2], 1)
    return (cv3).forward(argument_1, input, )
