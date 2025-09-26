class Bottleneck(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  cv1 : __torch__.models.common.___torch_mangle_189.Conv
  cv2 : __torch__.models.common.___torch_mangle_192.Conv
  def forward(self: __torch__.models.common.___torch_mangle_193.Bottleneck,
    argument_1: Tensor) -> Tensor:
    cv2 = self.cv2
    cv20 = self.cv2
    act = cv20.act
    cv1 = self.cv1
    _0 = (cv2).forward((cv1).forward(act, argument_1, ), )
    return _0
