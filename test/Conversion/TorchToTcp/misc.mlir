// RUN: torch-mlir-opt <%s -convert-torch-to-tcp -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @torch.aten.broadcast_to(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[1,?],f32>,
// CHECK-SAME:         %[[ARG1:.*]]: !torch.int,
// CHECK-SAME:         %[[ARG2:.*]]: !torch.int) -> !torch.vtensor<[?,?,?],f32> {
// CHECK:         %[[TO_BUILTIN:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[1,?],f32> -> tensor<1x?xf32>
// CHECK:         %[[EXPAND_SHAPE:.*]] = tensor.expand_shape %[[TO_BUILTIN]]
// CHECK-SAME:                           [0, 1], [2]
// CHECK-SAME:                           : tensor<1x?xf32> into tensor<1x1x?xf32>
// CHECK:         %[[ARG1_I64:.*]] = torch_c.to_i64 %[[ARG1]]
// CHECK:         %[[ARG1_INDEX:.*]] = arith.index_cast %[[ARG1_I64]] : i64 to index
// CHECK:         %[[ARG2_I64:.*]] = torch_c.to_i64 %[[ARG2]]
// CHECK:         %[[ARG2_INDEX:.*]] = arith.index_cast %[[ARG2_I64]] : i64 to index
// CHECK:         %[[BROADCAST:.*]] = tcp.broadcast %[[EXPAND_SHAPE]], %[[ARG1_INDEX]], %[[ARG2_INDEX]]
// CHECK-SAME:                        {axes = [0, 1]}
// CHECK-SAME:                        : tensor<1x1x?xf32>, index, index -> tensor<?x?x?xf32>
// CHECK:         %[[FROM_BUILTIN:.*]] = torch_c.from_builtin_tensor %[[BROADCAST]] : tensor<?x?x?xf32> -> !torch.vtensor<[?,?,?],f32>
// CHECK:         return %[[FROM_BUILTIN]] : !torch.vtensor<[?,?,?],f32>
func.func @torch.aten.broadcast_to(%arg0: !torch.vtensor<[1,?],f32>, %arg1: !torch.int, %arg2: !torch.int) -> !torch.vtensor<[?,?,?],f32> {
  %int-1 = torch.constant.int -1
  %0 = torch.prim.ListConstruct %arg1, %arg2, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.broadcast_to %arg0, %0 : !torch.vtensor<[1,?],f32>, !torch.list<int> -> !torch.vtensor<[?,?,?],f32>
  return %1 : !torch.vtensor<[?,?,?],f32>
}
