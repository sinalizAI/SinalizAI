class DetectionModel(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  model : __torch__.torch.nn.modules.container.___torch_mangle_199.Sequential
  def forward(self: __torch__.models.yolo.DetectionModel,
    x: Tensor) -> Tuple[Tensor]:
    model = self.model
    _24 = getattr(model, "24")
    model0 = self.model
    _23 = getattr(model0, "23")
    model1 = self.model
    _22 = getattr(model1, "22")
    model2 = self.model
    _21 = getattr(model2, "21")
    model3 = self.model
    _20 = getattr(model3, "20")
    model4 = self.model
    _19 = getattr(model4, "19")
    model5 = self.model
    _18 = getattr(model5, "18")
    model6 = self.model
    _17 = getattr(model6, "17")
    model7 = self.model
    _16 = getattr(model7, "16")
    model8 = self.model
    _15 = getattr(model8, "15")
    model9 = self.model
    _14 = getattr(model9, "14")
    model10 = self.model
    _13 = getattr(model10, "13")
    model11 = self.model
    _12 = getattr(model11, "12")
    model12 = self.model
    _11 = getattr(model12, "11")
    model13 = self.model
    _10 = getattr(model13, "10")
    model14 = self.model
    _9 = getattr(model14, "9")
    model15 = self.model
    _8 = getattr(model15, "8")
    model16 = self.model
    _7 = getattr(model16, "7")
    model17 = self.model
    _6 = getattr(model17, "6")
    model18 = self.model
    _5 = getattr(model18, "5")
    model19 = self.model
    _4 = getattr(model19, "4")
    model20 = self.model
    _3 = getattr(model20, "3")
    model21 = self.model
    _2 = getattr(model21, "2")
    model22 = self.model
    _1 = getattr(model22, "1")
    model23 = self.model
    _230 = getattr(model23, "23")
    m = _230.m
    _0 = getattr(m, "0")
    cv2 = _0.cv2
    act = cv2.act
    model24 = self.model
    _00 = getattr(model24, "0")
    _25 = (_1).forward(act, (_00).forward(act, x, ), )
    _26 = (_3).forward(act, (_2).forward(act, _25, ), )
    _27 = (_4).forward(act, _26, )
    _28 = (_6).forward(act, (_5).forward(act, _27, ), )
    _29 = (_8).forward(act, (_7).forward(act, _28, ), )
    _30 = (_10).forward(act, (_9).forward(act, _29, ), )
    _31 = (_12).forward((_11).forward(_30, ), _28, )
    _32 = (_14).forward(act, (_13).forward(act, _31, ), )
    _33 = (_16).forward((_15).forward(_32, ), _27, )
    _34 = (_17).forward(act, _33, )
    _35 = (_19).forward((_18).forward(act, _34, ), _32, )
    _36 = (_20).forward(act, _35, )
    _37 = (_22).forward((_21).forward(act, _36, ), _30, )
    _38 = (_24).forward(_34, _36, (_23).forward(_37, ), )
    return (_38,)
class Detect(Module):
  __parameters__ = []
  __buffers__ = ["anchors", ]
  anchors : Tensor
  training : bool
  _is_full_backward_hook : Optional[bool]
  m : __torch__.torch.nn.modules.container.ModuleList
  def forward(self: __torch__.models.yolo.Detect,
    argument_1: Tensor,
    argument_2: Tensor,
    argument_3: Tensor) -> Tensor:
    m = self.m
    _2 = getattr(m, "2")
    m0 = self.m
    _1 = getattr(m0, "1")
    m1 = self.m
    _0 = getattr(m1, "0")
    _14 = (_0).forward(argument_1, )
    bs = ops.prim.NumToTensor(torch.size(_14, 0))
    _15 = int(bs)
    _16 = int(bs)
    ny = ops.prim.NumToTensor(torch.size(_14, 2))
    _17 = int(ny)
    nx = ops.prim.NumToTensor(torch.size(_14, 3))
    _18 = torch.view(_14, [_16, 3, 25, _17, int(nx)])
    _19 = torch.contiguous(torch.permute(_18, [0, 1, 3, 4, 2]))
    _20 = torch.split_with_sizes(torch.sigmoid(_19), [2, 2, 21], 4)
    xy, wh, conf, = _20
    _21 = torch.add(torch.mul(xy, CONSTANTS.c0), CONSTANTS.c1)
    xy0 = torch.mul(_21, torch.select(CONSTANTS.c2, 0, 0))
    _22 = torch.pow(torch.mul(wh, CONSTANTS.c0), 2)
    wh0 = torch.mul(_22, CONSTANTS.c3)
    y = torch.cat([xy0, wh0, conf], 4)
    _23 = torch.mul(torch.mul(nx, CONSTANTS.c4), ny)
    _24 = torch.view(y, [_15, int(_23), 25])
    _25 = (_1).forward(argument_2, )
    bs0 = ops.prim.NumToTensor(torch.size(_25, 0))
    _26 = int(bs0)
    _27 = int(bs0)
    ny0 = ops.prim.NumToTensor(torch.size(_25, 2))
    _28 = int(ny0)
    nx0 = ops.prim.NumToTensor(torch.size(_25, 3))
    _29 = torch.view(_25, [_27, 3, 25, _28, int(nx0)])
    _30 = torch.contiguous(torch.permute(_29, [0, 1, 3, 4, 2]))
    _31 = torch.split_with_sizes(torch.sigmoid(_30), [2, 2, 21], 4)
    xy1, wh1, conf0, = _31
    _32 = torch.add(torch.mul(xy1, CONSTANTS.c0), CONSTANTS.c5)
    xy2 = torch.mul(_32, torch.select(CONSTANTS.c2, 0, 1))
    _33 = torch.pow(torch.mul(wh1, CONSTANTS.c0), 2)
    wh2 = torch.mul(_33, CONSTANTS.c6)
    y0 = torch.cat([xy2, wh2, conf0], 4)
    _34 = torch.mul(torch.mul(nx0, CONSTANTS.c4), ny0)
    _35 = torch.view(y0, [_26, int(_34), 25])
    _36 = (_2).forward(argument_3, )
    bs1 = ops.prim.NumToTensor(torch.size(_36, 0))
    _37 = int(bs1)
    _38 = int(bs1)
    ny1 = ops.prim.NumToTensor(torch.size(_36, 2))
    _39 = int(ny1)
    nx1 = ops.prim.NumToTensor(torch.size(_36, 3))
    _40 = torch.view(_36, [_38, 3, 25, _39, int(nx1)])
    _41 = torch.contiguous(torch.permute(_40, [0, 1, 3, 4, 2]))
    _42 = torch.split_with_sizes(torch.sigmoid(_41), [2, 2, 21], 4)
    xy3, wh3, conf1, = _42
    _43 = torch.add(torch.mul(xy3, CONSTANTS.c0), CONSTANTS.c7)
    xy4 = torch.mul(_43, torch.select(CONSTANTS.c2, 0, 2))
    _44 = torch.pow(torch.mul(wh3, CONSTANTS.c0), 2)
    wh4 = torch.mul(_44, CONSTANTS.c8)
    y1 = torch.cat([xy4, wh4, conf1], 4)
    _45 = torch.mul(torch.mul(nx1, CONSTANTS.c4), ny1)
    _46 = [_24, _35, torch.view(y1, [_37, int(_45), 25])]
    return torch.cat(_46, 1)
