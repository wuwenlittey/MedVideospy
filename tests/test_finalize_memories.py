import copy
import json
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tests.test_evolving import FakeReflector, base_config, write_source
from videospy.evolving.finalize_memories import finalize_memories, main
from videospy.evolving.pipeline import EvolutionPipeline
from videospy.evolving.storage.memory_store import EvolutionStateStore, hash_embedding


class FinalizeMemoriesTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'source'
        write_source(self.source, count=4)
        self.manifest = self.root / 'train.json'
        self.manifest.write_text('[0,1,2,3]')
        self.config = base_config()
        self.run = self.root / 'run'
        pipeline = EvolutionPipeline('lvbench', self.manifest, self.root, self.config,
                                     self.source, run_dir=self.run, reflector=FakeReflector())
        with patch('videospy.evolving.pipeline.MemoryOptimizer.optimize', side_effect=RuntimeError('interrupted')):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                pipeline.run()
        self.report_path = self.run / 'reflection/report.json'
        self.report = json.loads(self.report_path.read_text())
        self.failures = [{'stage': 'memory', 'node': 'original-failure', 'reason': 'too long'}]
        self.failures.extend({'stage': 'reasoning_case', 'node': str(i), 'reason': 'invalid reference'} for i in range(3))
        self.report['failures'] = copy.deepcopy(self.failures)
        self.report['stages']['memory']['failed'] = 1
        self.report['stages']['reasoning_case'].update(succeeded=1, failed=3)
        self.report_path.write_text(json.dumps(self.report))
        # The finalizer must not require access to the original training inputs.
        self.source.rename(self.root / 'unavailable-source')
        self.manifest.unlink()
        self.store = EvolutionStateStore(self.run / 'evolution.sqlite3', read_only=True)

    def test_partial_publication_without_reflector_and_idempotent_resume(self):
        with patch('videospy.evolving.reflector.Reflector.__init__', side_effect=AssertionError('reflector called')), \
             patch('videospy.model.ModelClient.chat', side_effect=AssertionError('chat called')), \
             patch('videospy.evolving.finalize_memories.embedding_function_from_config', return_value=hash_embedding):
            result = finalize_memories(self.run, self.config)
            self.assertEqual(result['status'], 'partial')
            self.assertEqual(result['failures'], self.failures)
            self.assertEqual(result['memory_count'], len(self.report['memories']))
            self.assertEqual(self.store.load_skills(result['version_id']), self.report['skills'])
            self.assertEqual(self.store.resolve_version(), result['version_id'])
        with patch('videospy.evolving.finalize_memories.embedding_function_from_config',
                   return_value=lambda text: self.fail('cached vectors should be reused')):
            again = finalize_memories(self.run, self.config)
        self.assertEqual(result['version_id'], again['version_id'])
        self.assertEqual(again['stages']['embedding']['reused'], result['memory_count'])
        with sqlite3.connect(self.run / 'evolution.sqlite3') as connection:
            self.assertEqual(connection.execute('select count(*) from evolution_versions').fetchone()[0], 2)

    def test_failed_embeddings_are_recorded_and_successful_vectors_resume(self):
        first = hash_embedding('first')
        with patch('videospy.evolving.finalize_memories.embedding_function_from_config') as factory:
            from unittest.mock import Mock
            embed = Mock(side_effect=[first] + [ConnectionError('offline')] * 3)
            factory.return_value = embed
            result = finalize_memories(self.run, self.config)
        self.assertEqual(embed.call_count, 4)
        self.assertEqual(result['stages']['embedding']['failed'], 3)
        self.assertEqual(result['memory_count'], 1)
        self.assertEqual(result['failures'][:4], self.failures)
        self.assertEqual(len(result['failures']), 7)
        self.assertEqual(self.store.run_metadata()['status'], 'partial')
        with patch('videospy.evolving.finalize_memories.embedding_function_from_config', return_value=hash_embedding):
            result = finalize_memories(self.run, self.config)
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['stages']['embedding']['reused'], 1)
        self.assertEqual(result['failures'], self.failures)
        self.assertEqual(result['memory_count'], len(self.report['memories']))

    def test_wrong_embedding_configuration_rejected_without_mutation(self):
        before = (self.run / 'evolution.sqlite3').read_bytes()
        config = copy.deepcopy(self.config)
        config['tools']['memory_search']['embedding_model'] = {'model_name': 'other-model'}
        with self.assertRaisesRegex(ValueError, 'Embedding .* differs'):
            finalize_memories(self.run, config)
        self.assertEqual((self.run / 'evolution.sqlite3').read_bytes(), before)

    def test_unapproved_skill_rejected(self):
        next(iter(self.report['skills'].values()))['review'][-1]['approved'] = False
        self.report_path.write_text(json.dumps(self.report))
        with self.assertRaisesRegex(ValueError, 'unapproved Skill'):
            finalize_memories(self.run, self.config)

    def test_cli_returns_nonzero_for_partial_but_writes_artifacts(self):
        with patch('videospy.evolving.finalize_memories.build_config', return_value=self.config):
            code = main(['--run-dir', str(self.run)])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads((self.run / 'final_version.json').read_text())['status'], 'partial')


if __name__ == '__main__':
    unittest.main()
