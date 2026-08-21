# Version 3 frozen method and claim policy

## Primary method

Global `rmsnorm_bound` allocation followed by within-layer
`rmsnorm_ellipsoid_bound` ranking, p95 aggregation across experts, alignment-16
packed same-channel structured removal, and physical tensor repacking.

## Supported wording

- “The ellipsoid score significantly improves matched perplexity-based channel
  selection across moderate pruning budgets.”
- “p95 expert aggregation provides a significantly better empirical quality
  trade-off than worst-expert max aggregation.”
- “The method produces physically smaller, reloadable checkpoints.”
- “Approximately 6% MoE-width reduction is the current empirical knee of the
  compression–accuracy curve.”

The expert-specific ellipsoid score is a mathematical upper bound. Maximum
aggregation is conservative across all experts for one packed shared channel.
p95 aggregation is an empirical cross-expert selection heuristic and is not a
uniform all-expert certificate.

## Prohibited wording pending further evidence

Do not describe the 2%, 4%, 6%, or 8% checkpoints as lossless or near-lossless.
Do not claim universal superiority. Do not claim inference acceleration until
end-to-end measurements and operator evidence show it. Do not compare pruning
percentages across methods until MoE parameters, whole-model parameters, active
FLOPs, and serialized bytes use reconciled denominators.

