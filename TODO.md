# TODO List

- [ ] Fix `_apply_filter` to use `axis=-1` to correctly handle multi-dimensional arrays (channels, samples).
- [ ] Verify `fs` scaling matches the intended filter cutoffs.
- [ ] Implement the Low-Pass Filter step in the chain.
- [ ] Add unit test for complex data handling after `mix`.
- [ ] Refactor `history` to support sequential ordering (e.g., using a list) while retaining named key access for specific steps.
