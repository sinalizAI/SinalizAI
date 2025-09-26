class Sequential(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  __annotations__["0"] = __torch__.models.common.Bottleneck
  def forward(self: __torch__.torch.nn.modules.container.Sequential,
    argument_1: __torch__.torch.nn.modules.activation.___torch_mangle_191.SiLU,
    argument_2: Tensor) -> Tensor:
    _0 = getattr(self, "0")
    _1 = (_0).forward(argument_1, argument_2, )
    return _1
class ModuleList(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  __annotations__["0"] = __torch__.torch.nn.modules.conv.___torch_mangle_196.Conv2d
  __annotations__["1"] = __torch__.torch.nn.modules.conv.___torch_mangle_197.Conv2d
  __annotations__["2"] = __torch__.torch.nn.modules.conv.___torch_mangle_198.Conv2d
