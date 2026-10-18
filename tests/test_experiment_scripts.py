"""Run real offline CLI/SQLite through the shell entry points with simulated model replies."""
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.test_evolving import base_config, write_source

ROOT=Path(__file__).resolve().parents[1]


class ManualExperimentScriptsTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        binary=self.root/'python'
        binary.write_text(f'#!{sys.executable}\n'+f'import sys\nsys.path.insert(0, {str(ROOT)!r})\n'+'''
import argparse, json, os, types
from pathlib import Path
from tests.test_evolving import base_config, fake_chat
from videospy.model import ModelClient
import config.videospy as config
# Use the same local embedding backend in training and evaluation without network calls.
config.default_config['tools']['memory_search'] = base_config()['tools']['memory_search']
def simulated_chat(self, messages, **kwargs):
    prompt = messages[0]['content']
    if prompt.startswith('Evaluate one completed'):
        data = json.JSONDecoder().raw_decode(prompt.split('\\nInput:\\n', 1)[1])[0]
        if os.environ.get('FAIL_CLASSIFICATION') in ('all', data['sample_id']):
            raise RuntimeError('injected classification failure')
    if prompt.startswith('Synthesize a concise') and os.environ.get('FAIL_DOMAIN'):
        raise RuntimeError('injected Domain reflection failure')
    return fake_chat(self, messages, **kwargs)
ModelClient.chat = simulated_chat
args = sys.argv[1:]
with open(os.environ['CALL_LOG'], 'a') as stream:
    stream.write(json.dumps(args) + '\\n')
if args[1] == 'videospy.evolving.cli':
    from videospy.evolving.cli import main
    raise SystemExit(main(args[2:]))
# Parse the real benchmark CLI and load its selected evolution components.
try:
    import decord
except ImportError:
    stub=types.ModuleType('decord');stub.VideoReader=lambda *a, **k:None
    sys.modules['decord']=stub
import importlib
runner=importlib.import_module(args[1])
parsed=runner.parse_args(args[2:])
loaded=config.build_config(parsed, getattr(parsed,'config_paths',[]))
expected_paths = [Path(os.environ['CONFIG'])] if os.environ.get('CONFIG') else []
expected = config.build_config(argparse.Namespace(), expected_paths)
assert loaded['agent']['max_steps'] == expected['agent']['max_steps']
assert loaded['agent']['model'] == expected['agent']['model']
for tool in ('overview', 'clip_skim', 'frame_inspect'):
    assert loaded['tools'][tool]['model'] == expected['tools'][tool]['model']
assert loaded['tools']['clip_skim']['num_frames'] == expected['tools']['clip_skim']['num_frames']
if parsed.evolution_db:
    if os.environ.get('FAIL_DOMAIN'):
        assert '__domain__' not in loaded['TASK_SKILLS']
        assert loaded['TASK_SKILLS'] or not loaded['_evolution']['components']['skill']
    else:
        assert '__domain__' in loaded['TASK_SKILLS'] or not loaded['_evolution']['components']['skill']
    assert loaded['tools']['memory_search']['enabled'] == loaded['_evolution']['components']['memory']
path=Path(parsed.run_dir);path.mkdir(parents=True,exist_ok=True)
runner.ensure_run_manifest(path, {'agent_config': loaded})
(path/'report.md').write_text('Simulated video evaluation; real CLI and evolution loading checked.')
raise SystemExit(int(os.environ.get('EVAL_STATUS','0')))
''')
        binary.chmod(0o755)

    def test_three_training_scripts_execute_classification_reflection_storage_and_evaluation_loading(self):
        import yaml
        for dataset in ('lvbench','egolife','medvidbench'):
            with self.subTest(dataset=dataset):
                root=self.root/dataset;root.mkdir()
                source=root/'source';write_source(source,dataset)
                manifest=root/'train.json';manifest.write_text(json.dumps(list(range(8))))
                training_config=base_config()
                training_config['agent']['max_steps']=7
                training_config['tools']['clip_skim']['num_frames']=6
                inference_model={'model_name':'openai/test-inference','api_base':'https://inference.example.test/v1'}
                training_config['agent']['model'].update(inference_model)
                for tool in ('overview','clip_skim','frame_inspect'):
                    training_config['tools'][tool]['model']=dict(inference_model)
                # Inference switches must not disable either training route.
                training_config['evolving']['inference']={'skill':False,'memory':False,'hyperparameter':False}
                cfg=root/'config.yml';cfg.write_text(yaml.safe_dump(training_config))
                log=root/'calls.jsonl'
                env={**os.environ,'PATH':str(self.root)+os.pathsep+os.environ['PATH'],
                     'MODE':'full','WORKERS':'3','CONFIG':str(cfg),'MANIFEST':str(manifest),'TEST_MANIFEST':str(manifest),
                     'INFERENCE_RUN_DIR':str(source),'OUTPUT_DIR':str(root/'out'),'RUN_DIR':'','CALL_LOG':str(log)}
                before={p:p.read_bytes() for p in source.rglob('*') if p.is_file()}
                result=subprocess.run(['bash',str(ROOT/f'experiments/train/{dataset}/train_offline.sh')],cwd=ROOT,env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
                calls=[json.loads(line) for line in log.read_text().splitlines()]
                self.assertEqual(len(calls),5)
                self.assertEqual(calls[0][:3],['-m','videospy.evolving.cli','run'])
                self.assertEqual(calls[0][calls[0].index('--config')+1],str(cfg))
                self.assertTrue(all(c[c.index('--workers')+1]=='3' for c in calls))
                run=next(root.rglob('final_version.json')).parent.resolve()
                self.assertEqual(run.parent, (root/'out').resolve())
                skills=json.loads((run/'skills.json').read_text());memories=json.loads((run/'memories.json').read_text())
                self.assertIn('__domain__',skills);self.assertGreater(len(memories),0)
                self.assertTrue(all(v['format']=='document' for v in skills.values()))
                self.assertTrue(all(m['metadata']['source']['video'] for m in memories))
                self.assertEqual(before,{p:p.read_bytes() for p in source.rglob('*') if p.is_file()})
                order = ('skill_only','base','memory_only','skill_memory') if dataset == 'lvbench' else ('skill_only','memory_only','skill_memory','base')
                for call,component in zip(calls[1:], order):
                    self.assertEqual(Path(call[call.index('--run-dir')+1]),run/'evaluations'/component/'full')
                    self.assertNotIn('--evolution-version',call)
                    configs=[call[i+1] for i,value in enumerate(call) if value=='--config']
                    loaded=json.loads((run/'evaluations'/component/'full'/'run.json').read_text())['agent_config']
                    if component!='base':
                        self.assertEqual(configs,[str(cfg),f'experiments/test/test_configs/{component}.yml'])
                        self.assertEqual(loaded['_evolution']['components'],{
                            'skill':component!='memory_only','memory':component!='skill_only','hyperparameter':False})
                        self.assertEqual(loaded['_evolution']['version_id'],json.loads((run/'offline_result.json').read_text())['version_id'])
                    else:
                        self.assertEqual(configs,[str(cfg)])
                        self.assertNotIn('--evolution-db',call)
                        self.assertNotIn('_evolution',loaded)
                        self.assertFalse(loaded['TASK_SKILLS'])
                        self.assertFalse(loaded['tools']['memory_search']['enabled'])
                # Resume the same CLI run and continue reporting evaluation errors after all four evaluations.
                env.update(RUN_DIR=str(run),EVAL_STATUS='8')
                resumed=subprocess.run(['bash',str(ROOT/f'experiments/train/{dataset}/train_offline.sh')],cwd=ROOT,env=env,capture_output=True,text=True)
                self.assertEqual(resumed.returncode,8,resumed.stdout+resumed.stderr)
                self.assertEqual(len(log.read_text().splitlines()),10)
                self.assertIn('Reusing unchanged offline snapshot',(run/'evolution.log').read_text())

                # Standalone defaults differ from the explicit training config and require a fresh run.
                evaluation_env={**env,'EVAL_STATUS':'0','CONFIG':'','OUTPUT_DIR':'',
                                'EVOLUTION_RUN_DIR':str(run),'EVOLUTION_DB':'','RUN_DIR':''}
                for component,suffix in (
                    ('base',''), ('skill_memory','_evolve'),
                ):
                    target=run/'evaluations'/component/'full'
                    before_manifest=(target/'run.json').read_bytes()
                    standalone_env=dict(evaluation_env)
                    if component=='base':
                        standalone_env['RUN_DIR']=str(target)
                    script=ROOT/f'experiments/test/{dataset}/test_W_testsplit{suffix}.sh'
                    result=subprocess.run(['bash',str(script)],cwd=ROOT,env=standalone_env,
                                          capture_output=True,text=True)
                    self.assertEqual(result.returncode,1,result.stdout+result.stderr)
                    self.assertIn('Run configuration does not match existing manifest',result.stderr)
                    self.assertEqual((target/'run.json').read_bytes(),before_manifest)

                    # Both entries also accept a new explicit directory after configuration changes.
                    fresh=root/'fresh evaluations'/component
                    standalone_env['RUN_DIR']=str(fresh)
                    result=subprocess.run(['bash',str(script)],cwd=ROOT,env=standalone_env,
                                          capture_output=True,text=True)
                    self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                    fresh_manifest=json.loads((fresh/'run.json').read_text())
                    expected_manifest=json.loads(before_manifest)
                    self.assertNotEqual(fresh_manifest['agent_config']['agent']['model'],
                                        expected_manifest['agent_config']['agent']['model'])

    def test_failed_training_stops_evaluations_and_changed_results_require_fresh_evaluation(self):
        import yaml
        source=self.root/'source';write_source(source)
        manifest=self.root/'train.json';manifest.write_text('[0,1,2,3]')
        cfg=self.root/'config.yml';cfg.write_text(yaml.safe_dump(base_config()))
        log=self.root/'calls.jsonl'
        env={**os.environ,'PATH':str(self.root)+os.pathsep+os.environ['PATH'],
             'MODE':'full','WORKERS':'3','CONFIG':str(cfg),'MANIFEST':str(manifest),'TEST_MANIFEST':str(manifest),
             'INFERENCE_RUN_DIR':str(source),'OUTPUT_DIR':str(self.root/'out'),'RUN_DIR':'',
             'CALL_LOG':str(log),'FAIL_CLASSIFICATION':'all'}
        command=['bash',str(ROOT/'experiments/train/lvbench/train_offline.sh')]
        failed=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True)
        self.assertEqual(failed.returncode,1,failed.stdout+failed.stderr)
        run=next((self.root/'out').rglob('offline_result.json')).parent
        env['RUN_DIR']=str(run)
        self.assertEqual(len(log.read_text().splitlines()),1)
        self.assertFalse((run/'evaluations').exists())
        self.assertGreater(len(json.loads((run/'memories.json').read_text())),0)
        env['FAIL_CLASSIFICATION']='0'
        partial=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True)
        self.assertEqual(partial.returncode,0,partial.stdout+partial.stderr)
        previous=json.loads((run/'offline_result.json').read_text())
        self.assertEqual(previous['status'],'partial')
        old_reports={path:path.read_bytes() for path in (run/'evaluations').rglob('report.md')}
        self.assertEqual(len(old_reports),4)
        env.pop('FAIL_CLASSIFICATION')
        recovered=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True)
        self.assertEqual(recovered.returncode,1,recovered.stdout+recovered.stderr)
        self.assertIn('Run configuration does not match existing manifest',recovered.stderr)
        current=json.loads((run/'offline_result.json').read_text())
        self.assertEqual(current['status'],'complete')
        self.assertNotEqual(current['evaluation_version'],previous['evaluation_version'])
        self.assertEqual(len(list((run/'evaluations').rglob('report.md'))),4)
        self.assertEqual(old_reports,{path:path.read_bytes() for path in old_reports})

    def test_domain_failure_allows_all_four_evaluations_in_three_original_scripts(self):
        import yaml
        for dataset in ('lvbench', 'egolife', 'medvidbench'):
            with self.subTest(dataset=dataset):
                root=self.root/dataset;root.mkdir()
                source=root/'source';write_source(source,dataset)
                manifest=root/'train.json';manifest.write_text(json.dumps(list(range(8))))
                cfg=root/'config.yml';cfg.write_text(yaml.safe_dump(base_config()))
                log=root/'calls.jsonl'
                env={**os.environ,'PATH':str(self.root)+os.pathsep+os.environ['PATH'],
                     'MODE':'full','WORKERS':'3','CONFIG':str(cfg),'MANIFEST':str(manifest),'TEST_MANIFEST':str(manifest),
                     'INFERENCE_RUN_DIR':str(source),'OUTPUT_DIR':str(root/'out'),'RUN_DIR':'',
                     'CALL_LOG':str(log),'FAIL_DOMAIN':'1'}
                result=subprocess.run(['bash',str(ROOT/f'experiments/train/{dataset}/train_offline.sh')],
                                      cwd=ROOT,env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
                self.assertEqual(len(log.read_text().splitlines()),5)
                run=next((root/'out').rglob('offline_result.json')).parent
                summary=json.loads((run/'offline_result.json').read_text())
                self.assertEqual(summary['status'],'partial')
                self.assertFalse(summary['domain_skill_available'])
                self.assertEqual(summary['stages']['domain_skill']['failed'],1)
                self.assertTrue(summary['evaluation_version'])
                self.assertEqual(len(list((run/'evaluations').rglob('report.md'))),4)

    def test_missing_source_and_invalid_mode_fail_before_python(self):
        for dataset in ('lvbench','egolife','medvidbench'):
            script=ROOT/f'experiments/train/{dataset}/train_offline.sh'
            for mode in ('invalid','full'):
                env={**os.environ,'MODE':mode,'INFERENCE_RUN_DIR':str(self.root/'missing')}
                result=subprocess.run(['bash',str(script)],cwd=ROOT,env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,2)

    def test_standalone_evolution_requires_completed_training(self):
        evolution=self.root/'incomplete evolution'
        evolution.mkdir()
        (evolution/'evolution.sqlite3').touch()
        log=self.root/'calls.jsonl'
        env={**os.environ,'PATH':str(self.root)+os.pathsep+os.environ['PATH'],
             'MODE':'full','EVOLUTION_RUN_DIR':str(evolution),'EVOLUTION_DB':'',
             'CALL_LOG':str(log)}
        for script in sorted((ROOT/'experiments/test').glob('*/*_evolve.sh')):
            with self.subTest(script=script.name,dataset=script.parent.name):
                result=subprocess.run(['bash',str(script)],cwd=ROOT,env=env,
                                      capture_output=True,text=True)
                self.assertEqual(result.returncode,2,result.stdout+result.stderr)
                self.assertIn('Evolution has not finished',result.stderr)
        self.assertFalse(log.exists())

    def test_deployment_header_defaults_and_overrides_without_launching_services(self):
        for name in ("aicloud_submit.sh", "qwen35_deploy.sh"):
            script = ROOT / "experiments" / name
            # Evaluate only parameter assignments, never conda, GPU probes or servers.
            assignments = re.findall(
                r"^(?:TASK_SCRIPT|MODEL_NAME|MODEL_PATH|SERVER_PORT|TP)=.*$",
                script.read_text(), re.M,
            )
            keys = [line.split("=", 1)[0] for line in assignments]
            for line, key in zip(assignments, keys):
                self.assertTrue(line.startswith(key + '="${' + key + ':-'))
            command = "\n".join(assignments)
            command += "\nprintf '%s\\n' " + " ".join(f'"${key}"' for key in keys)
            for override in (False, True):
                with self.subTest(script=name, override=override):
                    env = {**os.environ, **dict.fromkeys(keys, "")}
                    expected = {
                        "MODEL_NAME": "Qwen3.5-4B", "SERVER_PORT": "23333", "TP": "1",
                        "TASK_SCRIPT": "experiments/test/lvbench/test_W_trainsplit.sh",
                    }
                    if override:
                        expected = dict(MODEL_NAME="custom model", MODEL_PATH="/model with spaces",
                                        SERVER_PORT="24444", TP="2", TASK_SCRIPT="custom task.sh")
                        env.update(expected)
                    result = subprocess.run(["bash", "-eu", "-c", command], env=env, text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    values = dict(zip(keys, result.stdout.splitlines()))
                    for key in keys:
                        if key in expected:
                            self.assertEqual(values[key], expected[key])
                    if not override:
                        self.assertTrue(values["MODEL_PATH"].endswith("/models/Qwen/Qwen3.5-4B"))
