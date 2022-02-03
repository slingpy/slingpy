"""
Copyright (C) 2021  Patrick Schwab, Arash Mehrjou, GlaxoSmithKline plc; Andrew Jesson, University of Oxford; Ashkan Soleymani, MIT

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
 of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
import torch
from typing import List
from slingpy.transforms.abstract_transform import AbstractTransform


class ScaleTransform(AbstractTransform):
    def __init__(self, loc: List[torch.Tensor], scale: List[torch.Tensor]):
        super(ScaleTransform, self).__init__()
        self.loc = loc
        self.scale = scale

    def transform(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(self.loc) == len(inputs) and len(self.scale) == len(inputs), \
            "__loc__ and __scale__ in __ScaleTransform__ must be the same length as the inputs to be transformed."

        outs = []
        for input_i, loc_i, scale_i in zip(inputs, self.loc, self.scale):
            if isinstance(input_i, list):
                input_i = input_i[0]
            scaled = (input_i - loc_i) / scale_i
            outs.append(scaled)
        return outs

    def inverse_transform(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        outs = []
        for input_i, loc_i, scale_i in zip(inputs, self.loc, self.scale):
            if isinstance(input_i, list):
                input_i = input_i[0]
            scaled = input_i * scale_i + loc_i
            outs.append(scaled)
        return outs
