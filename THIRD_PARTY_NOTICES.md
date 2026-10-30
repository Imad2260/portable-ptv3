# Third-party notices

## Point Transformer V3 / Pointcept (MIT)

`ptv3/model.py`, `ptv3/attention.py` (RPE), `ptv3/data/transforms.py`, `ptv3/losses.py` and
`scripts/baselines/official.py` are derived from or reproduce the semantics of
[Pointcept/PointTransformerV3](https://github.com/Pointcept/PointTransformerV3) and
[Pointcept/Pointcept](https://github.com/Pointcept/Pointcept). Module and parameter names are kept so
official checkpoints load unchanged.

```
MIT License

Copyright (c) 2023 Pointcept

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The serialization codes reproduce Pointcept's z-order (from OCNN, Peng-Shuai Wang, MIT) and Hilbert
(from numpy-hilbert-curve, Princeton LIPS, MIT) encoders; `ptv3/serialization.py` computes the same
codes with integer bit operations (checked bit for bit against the official encoders).

## Data and weights

S3DIS (Armeni et al., CVPR 2016) is not redistributed here; `scripts/download.py` fetches Pointcept's
preprocessed copy, subject to the dataset's terms of use. The official PTv3 checkpoints come from
the Hugging Face Hub repository `Pointcept/PointTransformerV3`.
