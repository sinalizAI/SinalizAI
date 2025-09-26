class Sequential(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  __annotations__["0"] = __torch__.models.common.___torch_mangle_64.Bottleneck
  __annotations__["1"] = __torch__.models.common.___torch_mangle_71.Bottleneck
  __annotations__["2"] = __torch__.models.common.___torch_mangle_78.Bottleneck
  def forward(self: __torch__.torch.nn.modules.container.___torch_mangle_79.Sequential,
    argument_1: __torch__.torch.nn.modules.activation.___torch_mangle_191.SiLU,
    argument_2: Tensor) -> Tensor:
    _2 = getattr(self, "2")
    _1 = getattr(self, "1")
    _0 = getattr(self, "0")
    _3 = (_0).forward(argument_1, argument_2, )
    _4 = (_2).forward(argument_1, (_1).forward(argument_1, _3, ), )
    return _4
