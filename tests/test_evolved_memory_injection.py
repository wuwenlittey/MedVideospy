import copy
import sys
import types
import unittest
from unittest.mock import ANY, Mock, patch


try:
    import decord  # noqa: F401
except ModuleNotFoundError:
    decord_stub = types.ModuleType("decord")
    decord_stub.VideoReader = object
    sys.modules["decord"] = decord_stub

try:
    import litellm  # noqa: F401
except ModuleNotFoundError:
    litellm_stub = types.ModuleType("litellm")
    litellm_stub.completion = lambda **_: None
    sys.modules["litellm"] = litellm_stub


from videospy.agent import VideoSpyAgent
from videospy.evolved_config import apply_evolution_state
from videospy.evolving.storage.memory_store import embedding_metadata
from videospy.evolving.common.memory_scope import source_context


def agent_config() -> dict:
    return {
        "SYSTEM_PROMPT": (
            "overview={overview_num_frames}, skim={clip_skim_num_frames}, "
            "inspect={frame_inspect_max_frames}"
        ),
        "agent": {
            "max_steps": 2,
            "include_subtitles_in_question": True,
            "model": {},
        },
        "tools": {
            "enabled": [],
            "overview": {"num_frames": 4},
            "clip_skim": {"num_frames": 4},
            "frame_inspect": {"max_frames": 2},
            "memory_search": {
                "enabled": True,
                "database_path": "/tmp/evolution.sqlite3",
                "version_id": "run:final",
                "top_k": 3,
                "similarity_threshold": 0.4,
                "embedding_model": None,
            },
        },
        "evolving": {
            "inference": {
                "skill": True,
                "memory": True,
                "hyperparameter": True,
            }
        },
    }


class FakeVideoReader:
    duration = 10.0

    def __len__(self):
        return 100

    def get_avg_fps(self):
        return 10.0


class AutomaticMemoryInjectionTest(unittest.TestCase):
    def test_run_retrieves_and_injects_memories_without_a_tool_call(self):
        config = agent_config()
        memory_store = Mock()
        memory_store.search.return_value = [
            {
                "source_sample_id": "egolife-train-1", "memory_id":"m1",
                "metadata":{"source":{"source_id":"egolife:A1_JAKE"}},
                "context": "The question asks about a transition.",
                "content": "At DAY1 09:00 Jake introduced Alice as his sister.",
            }
        ]
        model_client = Mock()
        model_client.chat.side_effect = [
            {"role": "assistant", "content": "B"},
            {"role": "assistant", "content": "B"},
            {"role": "assistant", "content": "C"},
            {"role": "assistant", "content": "C"},
        ]

        with (
            patch("videospy.agent.ModelClient", return_value=model_client),
            patch(
                "videospy.evolving.storage.memory_store.MemoryStore",
                return_value=memory_store,
            ),
            patch(
                "videospy.evolving.storage.memory_store.embedding_function_from_config",
                return_value=Mock(),
            ),
        ):
            agent = VideoSpyAgent(
                config,
                "video.frames",
                None,
                "output",
                [],
                video_reader=FakeVideoReader(),
            )
        agent.subtitles = [
            {"start_time": 0, "end_time": 1, "subtitle": "Current evidence."}
        ]
        agent.task_types = ["retrieval", "reasoning"]
        agent.task_type = "retrieval|reasoning"

        trajectory = agent.run("What happens after the transition?")

        self.assertEqual(memory_store.search.call_count, 1)
        memory_store.search.assert_called_with(
            version_id="run:final",
            query="What happens after the transition?",
            task_types=None,
            context=None,
            audit=ANY,
            top_k=3,
            similarity_threshold=0.4,
        )
        first_prompt = model_client.chat.call_args_list[0].args[0][0]["content"]
        self.assertIn("# Prior Memory", first_prompt)
        self.assertIn("Source: egolife:A1_JAKE", first_prompt)
        self.assertIn("At DAY1 09:00 Jake introduced Alice as his sister.", first_prompt)
        self.assertIn("current observations take precedence", first_prompt)
        self.assertIn("not necessarily the latest state", first_prompt)
        self.assertEqual(model_client.chat.call_args_list[0].kwargs["tools"], [])
        self.assertEqual(trajectory.final_answer, "B")
        self.assertEqual(trajectory.to_dict()['diagnostics']['memory']['ids'], ['m1'])
        self.assertEqual(trajectory.to_dict()['diagnostics']['visual_calls']['executed'], 0)

        memory_store.search.return_value[0].pop("source_sample_id")
        second = agent.run("Where does the transition end?")
        second_prompt = model_client.chat.call_args_list[2].args[0][0]["content"]
        self.assertNotIn("Source sample:", second_prompt)
        self.assertEqual(second_prompt.count("# Prior Memory"), 1)
        self.assertEqual(memory_store.search.call_count, 2)
        self.assertEqual(second.final_answer, "C")

    def test_evolution_loading_removes_memory_search_from_enabled_tools(self):
        config = agent_config()
        config["tools"]["enabled"] = ["overview", "memory_search"]
        metadata = embedding_metadata(config["tools"]["memory_search"])
        store = Mock()
        store.resolve_version.return_value = "run:final"
        store.load_hyperparameters.return_value = {}
        store.load_skills.return_value = {}
        store.count_memories.return_value = 1
        store.run_metadata.return_value = {
            "embedding_backend": metadata["backend"],
            "embedding_model_name": metadata["model_name"],
            "embedding_config_hash": metadata["config_hash"],
        }
        store.version_metadata.return_value = {}

        with (
            patch("videospy.evolved_config.EvolutionStateStore", return_value=store),
            patch("videospy.evolved_config._sha256", return_value="digest"),
        ):
            loaded = apply_evolution_state(
                copy.deepcopy(config), "/tmp/evolution.sqlite3", "run:final"
            )

        self.assertTrue(loaded["tools"]["memory_search"]["enabled"])
        self.assertNotIn("memory_search", loaded["tools"]["enabled"])


class DomainAndMemoryTest(unittest.TestCase):
    def test_domain_is_loaded_once_including_precomposed_medvid_tasks(self):
        for precomposed in (False, True):
            config=agent_config()
            config['tools']['memory_search']['enabled']=False
            config['TASK_SKILLS']={
                '__domain__':{'format':'document','instruction':'DOMAIN METHOD','sources':[],'source_tasks':[],'review':[]},
                'retrieval':{'format':'document','instruction':'TASK METHOD','sources':[],'source_tasks':[],'review':[]}}
            config['_runtime']={'task_types':['retrieval'],'prompt_composed':precomposed}
            if precomposed:
                from benchmark.medvidbench.run import build_task_agent_config
                config=build_task_agent_config(config,'retrieval','videospy')
            with patch('videospy.agent.ModelClient'):
                agent=VideoSpyAgent(config,'video.frames',None,'output',[],video_reader=FakeVideoReader())
            prompt=agent.construct_initial_messages()[0]['content']
            self.assertEqual(prompt.count('DOMAIN METHOD'),1)
            self.assertEqual(prompt.count('TASK METHOD'),1)
            self.assertIn('Domain:',prompt)

    def test_memory_metadata_is_visible_and_retrieval_failure_is_nonfatal(self):
        config=agent_config()
        memory=Mock()
        memory.search.return_value=[{'source_sample_id':'s1',
            'context':'What happened?','content':'An observed event.',
            'metadata':{'source':{'source_id':'egolife:video-17','video':'video-17','query_time':{'date':'DAY1','time':'10000000'},
                                 'clip':{'first_frame':'0195.jpg','fps':0.5},
                                 'time_basis':'seconds from the start of this frame sequence'},
                        'time':{'label':'DAY1 09:00'},'event':'a meeting','uncertainty':'person is partly obscured'}}]
        with patch('videospy.agent.ModelClient'),patch('videospy.evolving.storage.memory_store.MemoryStore',return_value=memory):
            agent=VideoSpyAgent(config,'video.frames',None,'output',[],video_reader=FakeVideoReader())
        agent.memory_context=source_context('egolife','/timeline/A1_JAKE',{'date':'DAY1','time':'11000000'})
        prompt=agent._retrieve_memory_prompt('What happened?')
        self.assertEqual(memory.search.call_args.kwargs['context'],agent.memory_context)
        for value in ('video-17','DAY1 09:00','10000000','person is partly obscured'):self.assertIn(value,prompt)
        for value in ('Source clip', '0195.jpg'):self.assertNotIn(value,prompt)
        memory.search.side_effect=RuntimeError('temporary embedding error')
        self.assertEqual(agent._retrieve_memory_prompt('Question'),'')

    def test_memory_separates_video_positions_event_times_and_availability(self):
        agent = VideoSpyAgent.__new__(VideoSpyAgent)
        agent.config = agent_config()
        agent.memory_context = None
        agent.memory_store = Mock()
        metadata = {'source': {'source_id': 'egolife:A1_JAKE',
                              'query_time': {'date': 'DAY2', 'time': '12000000'}},
                    'time': {'start_seconds': 123456.789, 'end_seconds': 123460.123,
                             'label': 'DAY1 09:00 - DAY1 10:00',
                             'calendar': {'continuous': False}}}
        agent.memory_store.search.return_value = [
            {'memory_id': 'known', 'content': 'A local observation.', 'metadata': metadata},
            {'memory_id': 'unknown', 'content': 'Another observation.',
             'metadata': {'time': {'start_seconds': None, 'end_seconds': None, 'label': ''},
                          'evidence_frames': [12.]}}]
        prompt = agent._retrieve_memory_prompt('What happened?')
        self.assertIn('Video position: start=123456.789s, end=123460.123s', prompt)
        self.assertIn('Event time: DAY1 09:00 - DAY1 10:00 (recording gaps)', prompt)
        self.assertIn("Available from: {'date': 'DAY2', 'time': '12000000'}", prompt)
        self.assertEqual(prompt.count('Video position:'), 1)
        self.assertIn('Source inspection frames (event time unknown): 12.0s', prompt)
        self.assertNotIn('None', prompt)
        self.assertEqual(agent.memory_audit['ids'], ['known', 'unknown'])
