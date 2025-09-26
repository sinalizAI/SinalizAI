class C3(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  cv1 : __torch__.models.common.___torch_mangle_180.Conv
  cv2 : __torch__.models.common.___torch_mangle_183.Conv
  cv3 : __torch__.models.common.___torch_mangle_186.Conv
  m : __torch__.torch.nn.modules.container.___torch_mangle_194.Sequential
  def forward(self: __torch__.models.common.___torch_mangle_195.C3,
    argument_1: Tensor) -> Tensor:
    cv3 = self.cv3
    cv2 = self.cv2
    m = self.m
    m0 = self.m
    _0 = getattr(m0, "0")
    cv20 = _0.cv2
    act = cv20.act
    cv1 = self.cv1
    _1 = (m).forward((cv1).forward(act, argument_1, ), )
    _2 = [_1, (cv2).forward(act, argument_1, )]
    input = torch.cat(_2, 1)
    return (cv3).forward(act, input, )
