class Conv(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  conv : __torch__.torch.nn.modules.conv.Conv2d
  act : __torch__.torch.nn.modules.activation.SiLU
  def forward(self: __torch__.models.common.Conv,
    argument_1: __torch__.torch.nn.modules.activation.___torch_mangle_191.SiLU,
    x: Tensor) -> Tensor:
    conv = self.conv
    _0 = (argument_1).forward((conv).forward(x, ), )
    return _0
class C3(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  cv1 : __torch__.models.common.___torch_mangle_5.Conv
  cv2 : __torch__.models.common.___torch_mangle_8.Conv
  cv3 : __torch__.models.common.___torch_mangle_11.Conv
  m : __torch__.torch.nn.modules.container.Sequential
  def forward(self: __torch__.models.common.C3,
    argument_1: __torch__.torch.nn.modules.activation.___torch_mangle_191.SiLU,
    argument_2: Tensor) -> Tensor:
    cv3 = self.cv3
    cv2 = self.cv2
    m = self.m
    cv1 = self.cv1
    _1 = (cv1).forward(argument_1, argument_2, )
    _2 = (m).forward(argument_1, _1, )
    _3 = (cv2).forward(argument_1, argument_2, )
    input = torch.cat([_2, _3], 1)
    return (cv3).forward(argument_1, input, )
class Bottleneck(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  cv1 : __torch__.models.common.___torch_mangle_14.Conv
  cv2 : __torch__.models.common.___torch_mangle_17.Conv
  def forward(self: __torch__.models.common.Bottleneck,
    argument_1: __torch__.torch.nn.modules.activation.___torch_mangle_191.SiLU,
    argument_2: Tensor) -> Tensor:
    cv2 = self.cv2
    cv1 = self.cv1
    _4 = (cv1).forward(argument_1, argument_2, )
    _5 = torch.add(argument_2, (cv2).forward(argument_1, _4, ))
    return _5
class SPPF(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  cv1 : __torch__.models.common.___torch_mangle_104.Conv
  cv2 : __torch__.models.common.___torch_mangle_107.Conv
  m : __torch__.torch.nn.modules.pooling.MaxPool2d
  def forward(self: __torch__.models.common.SPPF,
    argument_1: __torch__.torch.nn.modules.activation.___torch_mangle_191.SiLU,
    argument_2: Tensor) -> Tensor:
    cv2 = self.cv2
    m = self.m
    cv1 = self.cv1
    _6 = (cv1).forward(argument_1, argument_2, )
    _7 = (m).forward(_6, )
    _8 = (m).forward1(_7, )
    input = torch.cat([_6, _7, _8, (m).forward2(_8, )], 1)
    return (cv2).forward(argument_1, input, )
class Concat(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.models.common.Concat,
    argument_1: Tensor,
    argument_2: Tensor) -> Tensor:
    input = torch.cat([argument_1, argument_2], 1)
    return input
