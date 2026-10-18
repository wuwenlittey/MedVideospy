import logging
from dataclasses import asdict

from tqdm import tqdm

from ..storage.artifacts import hash_json
from ..storage.memory_store import CachedEmbedding, cosine_similarity, validate_embedding
from ..common.schemas import MemoryEntry, empty_stage


def _normalized(text):
    return ' '.join(str(text or '').casefold().split())


def _same_source(left, right):
    a, b = left.get('source') or {}, right.get('source') or {}
    identity = a.get('source_id') or a.get('video')
    return bool(identity and a.get('benchmark') == b.get('benchmark')
                and identity == (b.get('source_id') or b.get('video')))


def _overlap(left, right):
    source_a, source_b = left.get('source') or {}, right.get('source') or {}
    if source_a.get('benchmark') == 'egolife' and (
        not source_a.get('timeline_id') or source_a['timeline_id'] != source_b.get('timeline_id')
    ):
        return None
    a, b = left.get('time') or {}, right.get('time') or {}
    bounds = (a.get('start_seconds'), a.get('end_seconds'), b.get('start_seconds'), b.get('end_seconds'))
    if any(value is None for value in bounds):
        return None
    return max(bounds[0], bounds[2]) <= min(bounds[1], bounds[3])


def _order(candidate):
    source = candidate['source']
    available = source.get('available_at')
    return (available if available is not None else float('inf'), source['sample_id'])


class MemoryOptimizer:
    def __init__(self, dedup_threshold=0.95, embedding_function=None):
        self.dedup_threshold = float(dedup_threshold)
        if not 0 <= self.dedup_threshold <= 1:
            raise ValueError('memory_dedup_threshold must be between 0 and 1')
        self.embedding_function = embedding_function
        self.logger = logging.getLogger(__name__)

    def set_logger(self, logger):
        self.logger = logger

    def optimize(self, store, source_version, report):
        embed = self.embedding_function or store.embedding_function
        existing = store.load_memories(source_version)
        accepted, duplicates = [], 0
        seen_sources = {(r['source_sample_id'], r['metadata'].get('kind', 'observation')) for r in existing}
        dimension = store.run_metadata()['embedding_dimension']
        stats = empty_stage()
        reused_before = embed.reused if isinstance(embed, CachedEmbedding) else 0
        for candidate in tqdm(sorted(report.memories, key=_order), desc="Memory deduplication", unit="memory", dynamic_ncols=True):
            source = candidate['source']
            sid = source['sample_id']
            kind = candidate.get('kind', 'observation')
            metadata = {'kind': kind, 'source': source, 'time': candidate['time'], 'event': candidate['event'],
                        'evidence_frames': candidate.get('evidence_frames', []),
                        'uncertainty': candidate.get('uncertainty', ''),
                        'task_types': candidate['task_types'], 'evidence_refs': candidate['evidence_refs']}
            if kind == 'reasoning_case':
                metadata['review'] = candidate['review']
            entry = MemoryEntry(
                id='memory:' + hash_json([source, candidate['content']] + ([kind] if kind == 'reasoning_case' else []))[:24],
                task_type=candidate['task_types'][0], context=candidate['context'],
                content=candidate['content'], source_sample_id=sid, metadata=metadata,
            )
            # A later source must never suppress an earlier usable memory.
            comparable = [row for row in existing if row['metadata'].get('kind', 'observation') == kind
                          and _same_source(row['metadata'], metadata)
                          and (source.get('benchmark') != 'egolife' or source.get('available_at') is None
                               or (row['metadata']['source'].get('available_at') is not None
                                   and row['metadata']['source']['available_at'] <= source['available_at']))]
            exact = any(_normalized(row['content']) == _normalized(entry.content)
                        and (_overlap(row['metadata'], metadata) is True
                             or (_overlap(row['metadata'], metadata) is None
                                 and _normalized(row['metadata'].get('event')) == _normalized(metadata['event'])))
                        for row in comparable)
            if (sid, kind) in seen_sources or exact:
                duplicates += 1
                stats['skipped'] += 1
                continue
            try:
                entry.embedding = validate_embedding(embed(store.memory_embedding_text(entry)), dimension)
                stats['succeeded'] += 1
            except Exception as error:
                self.logger.warning('Memory embedding failed; skipping sample %s: %s', sid, error)
                stats['failed'] += 1
                report.failures.append({'stage': 'embedding', 'node': sid, 'reason': str(error)[:1000]})
                continue
            if any(_overlap(row['metadata'], metadata) is True
                   and cosine_similarity(entry.embedding, row['embedding']) >= self.dedup_threshold
                   for row in comparable):
                duplicates += 1
                continue
            accepted.append(entry)
            existing.append(asdict(entry))
            seen_sources.add((sid, kind))
        if isinstance(embed, CachedEmbedding):
            stats['reused'] = embed.reused - reused_before
        report.stages['embedding'] = stats
        self.logger.info('Memory deduplication: proposed=%d added=%d duplicates=%d failed=%d',
                         len(report.memories), len(accepted), duplicates, stats['failed'])
        return accepted, {'added': len(accepted), 'duplicates': duplicates, 'failed': stats['failed']}
