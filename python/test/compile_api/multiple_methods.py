# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

import torch
import torch_mlir


class TwoMethodsModule(torch.nn.Module):
    def sin(self, x):
        return torch.ops.aten.sin(x)

    def cos(self, x):
        return torch.ops.aten.cos(x)


example_args = torch_mlir.ExampleArgs()
example_args.add_method("sin", torch.ones(2, 3))
example_args.add_method("cos", torch.ones(2, 4))

# Note: Due to https://github.com/pytorch/pytorch/issues/88735 we need to
# check the `use_tracing` case first.

print(torch_mlir.compile(TwoMethodsModule(), example_args, use_tracing=True))
# CHECK: module
# CHECK-DAG: func.func @sin
# CHECK-DAG: func.func @cos

print(torch_mlir.compile(TwoMethodsModule(), example_args))
# CHECK: module
# CHECK-DAG: func.func @sin
# CHECK-DAG: func.func @cos
