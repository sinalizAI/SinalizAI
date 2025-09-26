class Conv(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  conv : __torch__.torch.nn.modules.conv.___torch_mangle_190.Conv2d
  act : __torch__.torch.nn.modules.activation.___torch_mangle_191.SiLU
  def forward(self: __torch__.models.common.___torch_mangle_192.Conv,
    argument_1: Tensor) -> Tensor:
    act = self.act
    conv = self.conv
    _0 = (act).forward54((conv).forward(argument_1, ), )
    return _0
