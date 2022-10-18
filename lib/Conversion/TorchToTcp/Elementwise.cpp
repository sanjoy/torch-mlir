//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToTcp/TorchToTcp.h"

#include "PopulatePatterns.h"
#include "Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpDialect.h"
#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpOps.h"

using namespace mlir;
using namespace mlir::tcp;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {

bool skipMultiplyAlpha(Value alphaValue) {
  double doubleValue;
  if (matchPattern(alphaValue, m_TorchConstantFloat(&doubleValue)))
    return doubleValue == 1.0;

  int64_t intValue;
  if (matchPattern(alphaValue, m_TorchConstantInt(&intValue)))
    return intValue == 1;

  return false;
}

template <typename AtenOpT, typename TcpOpT>
class ConvertAtenAddOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;

  LogicalResult matchAndRewrite(
        AtenOpT op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.self();
    RankedTensorType lhsType = lhs.getType().dyn_cast<RankedTensorType>();
    auto lhsRank = lhsType.getRank();

    Value rhs = adaptor.other();
    RankedTensorType rhsType = rhs.getType().dyn_cast<RankedTensorType>();
    auto rhsRank = rhsType.getRank();

    if (!lhsType || !rhsType)
      return op.emitError("Only Ranked Tensor types are supported in TCP");

    if (lhsRank < rhsRank) {
      int64_t rankIncrease = rhsRank - lhsRank;
      lhs = torch_to_tcp::broadcastRankInLeadingDims(rewriter, lhs, rankIncrease);
      lhs = torch_to_tcp::broadcastShapeInLeadingDims(rewriter, lhs, rhs, rankIncrease);
    }
    if (lhsRank > rhsRank) {
      int64_t rankIncrease = lhsRank - rhsRank;
      rhs = torch_to_tcp::broadcastRankInLeadingDims(rewriter, rhs, rankIncrease);
      rhs = torch_to_tcp::broadcastShapeInLeadingDims(rewriter, rhs, lhs, rankIncrease);
    }

    if (!skipMultiplyAlpha(op.alpha()))
      return op.emitError("torch.add with alpha != 1 is not yet supported in Torch to TCP conversion");

    RankedTensorType resultType = OpConversionPattern<AtenOpT>::getTypeConverter()
                          ->convertType(op.getType())
                          .template cast<RankedTensorType>();
    rewriter.replaceOpWithNewOp<TcpOpT>(op, resultType, lhs, rhs);
    return success();
  }
};

template <typename AtenOpT, typename TcpOpT>
class ConvertAtenTanhOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;

  LogicalResult matchAndRewrite(
        AtenOpT op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.self();
    RankedTensorType inputType = input.getType().dyn_cast<RankedTensorType>();
    if (!inputType)
      return op.emitError("Only Ranked Tensor types are supported in TCP");
    if (!inputType.getElementType().isa<mlir::FloatType>())
      return op.emitError("Tanh input tensor must have floating-point datatype");

    rewriter.replaceOpWithNewOp<TcpOpT>(op, inputType, input);
    return success();
  }
};

} // namespace

void torch_to_tcp::populateElementwisePatternsAndLegality(TypeConverter &typeConverter,
                                            RewritePatternSet &patterns,
                                            ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();

  target.addIllegalOp<AtenTanhOp>();
  patterns.add<ConvertAtenTanhOp<AtenTanhOp, tcp::TanhOp>>(typeConverter, context);

  target.addIllegalOp<AtenAddTensorOp>();
  patterns.add<ConvertAtenAddOp<AtenAddTensorOp, tcp::AddOp>>(typeConverter, context);
}
