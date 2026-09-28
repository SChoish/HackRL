# Third-party notices

## Craftax_Baselines

Parts of `src/hackrl/ppo.py` are adapted from
[MichaelTMatthews/Craftax_Baselines](https://github.com/MichaelTMatthews/Craftax_Baselines)
at commit `7ce36fa05b84a2c9e758012f1e6da402e1e3a891`.

Preserved components: the symbolic actor-critic architecture, GAE recursion,
clipped PPO actor/value objectives, minibatch update ordering, orthogonal
initialization, gradient clipping, Adam settings, and learning-rate schedule.

HackRL-specific changes: environment injection, independent worker auto-reset,
terminal/reset observation preservation, task metrics, deterministic evaluation,
and the bounded pilot interface.

Copyright (c) 2024 Michael Matthews

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
