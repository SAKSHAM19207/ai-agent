# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Jaoe environment server components."""

__all__ = ["JaoeEnvironment"]


def __getattr__(name: str):
    if name == "JaoeEnvironment":
        from .jaoe_environment import JaoeEnvironment

        return JaoeEnvironment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
