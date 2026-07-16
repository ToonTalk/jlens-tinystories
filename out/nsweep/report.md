# n-sweep: how few fitting prompts does the 110M lens need?

Filtered top-8 agreement with the n=1000 lens at the last position of the 12 eval prompts, per mid-band layer; plus mean J cosine over L4-L8 and fit wall-clock.

| n | L4 | L5 | L6 | L7 | L8 | mean | J cos | fit wall-clock |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.28 | 0.36 | 0.44 | 0.54 | 0.72 | **0.47** | 0.8918 | 4 s |
| 5 | 0.49 | 0.55 | 0.62 | 0.71 | 0.76 | **0.63** | 0.9586 | 16 s |
| 10 | 0.57 | 0.62 | 0.70 | 0.71 | 0.79 | **0.68** | 0.9668 | 42 s |
| 25 | 0.58 | 0.65 | 0.67 | 0.72 | 0.80 | **0.68** | 0.9705 | 131 s |
| 100 | 0.78 | 0.82 | 0.82 | 0.88 | 0.92 | **0.84** | 0.9949 | 515 s |
| 1000 (ref) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** | 1.0000 | 6327 s |

Slice pages per n for eval prompts 4 and 9: `n****_prompt04/09.html`.
