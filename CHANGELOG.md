## Changelog

### Unreleased
- Breaking: `MetaQueryPlan::meta_filter` now returns `Self` instead of `Result<Self, String>`.
	- Any compile error from the expression is deferred and returned by `collect()`.
	- This removes the need to write `?` after `meta_filter()` in method chains; `?` after `collect()` is sufficient.

### 0.1.0-alpha1 - 2025-09-08
- Initial pre-release.
- SIMD-accelerated vector search with metadata filtering (MetaStore + VecStore).
- Early-stage: frequent breaking changes expected.
