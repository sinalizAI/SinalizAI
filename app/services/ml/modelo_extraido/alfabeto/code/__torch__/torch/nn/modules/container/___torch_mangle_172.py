class Sequential(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  __annotations__["0"] = __torch__.models.common.___torch_mangle_171.Bottleneck
  def forward(self: __torch__.torch.nn.modules.container.___torch_mangle_172.Sequential,
    argument_1: __torch__.torch.nn.modules.activation.___torch_mangle_191.SiLU,
    argument_2: Tensor) -> Tensor:
    _0 = getattr(self, "0")
    _1 = (_0).forward(argument_1, argument_2, )
    return _1
