class Conv(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  conv : __torch__.torch.nn.modules.conv.___torch_mangle_90.Conv2d
  act : __torch__.torch.nn.modules.activation.___torch_mangle_91.SiLU
  def forward(self: __torch__.models.common.___torch_mangle_92.Conv,
    argument_1: __torch__.torch.nn.modules.activation.___torch_mangle_191.SiLU,
    input: Tensor) -> Tensor:
    conv = self.conv
    _0 = (argument_1).forward30((conv).forward(input, ), )
    return _0
