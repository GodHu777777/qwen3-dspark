"""Explicit target cache lifecycle strategies for packed sampling.

Finite buckets describe storage applicability only, never scheduler capacity or
permission to use future confidence. Query lengths are exact and ordered.
"""
from dataclasses import dataclass

from .persistent_target_kv import Bucket, QueryFamily


class AppendCropTargetStrategy:
    """The original append/crop path, including its numerical backend."""
    def __init__(self, target):
        self.target = target

    def preflight(self, phase, query_lengths, context_lengths):
        pass

    def prefill(self, chunks):
        return self.target.append(chunks)

    def verify(self, chunks):
        return self.target.append(chunks)

    def commit(self, features, counts):
        for request, count in counts.items():
            self.target.crop(request, features.spans[request][2] + count)

    def invalidate(self):
        self.target.reset()

    def release_features(self, features):
        pass


@dataclass(frozen=True)
class FiniteTargetBuckets:
    """Caller supplies both finite lists; no implicit shape enumeration/padding."""
    prefill: tuple
    verification: tuple

    def __post_init__(self):
        for buckets in (self.prefill, self.verification):
            if (not isinstance(buckets, tuple) or not buckets or
                    any(not isinstance(b, (Bucket,QueryFamily)) for b in buckets) or
                    len(set(buckets)) != len(buckets)):
                raise ValueError('Explicit nonempty tuples of distinct buckets required')

    def select(self, phase, query_lengths, context_lengths):
        if phase not in ('prefill', 'verification'):
            raise ValueError('Unknown target bucket phase')
        query_lengths, context_lengths = tuple(query_lengths), tuple(context_lengths)
        if (len(query_lengths) != len(context_lengths) or
                any(type(q) is not int or q < 1 for q in query_lengths) or
                any(type(c) is not int or c < 0 for c in context_lengths)):
            raise ValueError('Matching nonnegative context lengths required')
        candidates = getattr(self, phase)
        compatible = [b for b in candidates if b.accepts(query_lengths,context_lengths)]
        if not compatible:
            raise ValueError(f'No finite {phase} bucket for ordered Q={query_lengths}, C={context_lengths}')
        # Stable declaration order breaks equal-capacity ties.
        return min(compatible, key=lambda b: b.key_capacity)


class PersistentTargetStrategy:
    """Opt-in eager CPU path: scratch verify, then one joint target commit."""
    def __init__(self, target, buckets):
        from .persistent_qwen_target import PersistentQwenTarget
        if not isinstance(target, PersistentQwenTarget) or not isinstance(buckets, FiniteTargetBuckets):
            raise ValueError('Persistent target and explicit finite bucket provider required')
        if target.lengths:
            raise ValueError('Strategy requires an initially empty persistent target')
        self.target, self.buckets = target, buckets
        self._features = None
        declared=tuple(dict.fromkeys(buckets.prefill + buckets.verification))
        families=tuple(b for b in declared if isinstance(b,QueryFamily))
        if families and not target.pool._families:target.register_families(families)
        for bucket in declared:
            target.register_bucket(bucket)

    def preflight(self, phase, query_lengths, context_lengths):
        return self.buckets.select(phase, query_lengths, context_lengths)

    def _run(self, phase, chunks):
        lengths = self.target.lengths
        bucket = self.preflight(phase, tuple(t.shape[1] for t in chunks.values()),
                                tuple(lengths[r] for r in chunks))
        operation = self.target.prefill if phase == 'prefill' else self.target.verify
        features = operation(chunks, bucket=bucket)
        self._features = features
        features.work.update(target_strategy='persistent', bucket_phase=phase,
            bucket_source=bucket.source,
            bucket_context_ceilings=list(bucket.context_ceilings) if isinstance(bucket,Bucket) else None,
            declared_phase_bucket_count=len(getattr(self.buckets, phase)),
            bucket_selection=('smallest_compatible_key_capacity_actual_physical_query_family'
                              if isinstance(bucket,QueryFamily) else 'smallest_compatible_key_capacity_exact_ordered_query'))
        return features

    def prefill(self, chunks):
        return self._run('prefill', chunks)

    def verify(self, chunks):
        return self._run('verification', chunks)

    def commit(self, features, counts):
        self.target.commit(features, counts)
        features.work.update(session_target_commit_calls=1,
                             committed_query_prefixes_including_anchor=dict(counts))

    def invalidate(self):
        # Also handles failure inside prefill after verify but before commit.
        # A poisoned store refuses abort/reset; the session still fails closed.
        errors = []
        pending = self.target._pending
        features = self._features if self._features is not None else (pending[0] if pending is not None else None)
        if pending is not None:
            try:self.target.abort(pending[0])
            except Exception as error:errors.append(error)
        if features is not None:
            try:self.release_features(features)
            except Exception as error:errors.append(error)
        try:self.target.reset()
        except Exception as error:errors.append(error)
        if errors:
            invalidate = getattr(self.target, 'invalidate', None)
            if invalidate is not None:
                try:invalidate()
                except Exception as error:errors.append(error)
            raise RuntimeError('Persistent target cleanup failed: ' + '; '.join(map(str,errors))) from errors[0]

    def release_features(self, features):
        # Eager CPU adapters predating graph leases have no release method.
        release = getattr(self.target, 'release_features', None)
        if release is not None:release(features)
        if self._features is features:self._features = None
