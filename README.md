# stocksift

## Purpose
- Identify relatively attractive long candidates within a stock universe.
- Support stock selection rather than predict overall market direction.

## Approach
- Generate simple historical valuation, fundamental, momentum, and risk features.
- Filter out clearly unattractive candidates.
- Rank remaining candidates using configurable scoring rules.

## Design Principles
- Keep assessment, selection policy, and evaluation separate.

## Limitations
- Not intended to prove statistically significant alpha.
- Historical results are used to assess robustness rather than provide statistical proof.
- Selection filters do not guarantee lower portfolio risk.
