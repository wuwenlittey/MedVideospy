"""Textual Skills do not install execution policies; baseline guards remain active."""
from unittest.mock import Mock

from tests.test_evolved_memory_injection import FakeVideoReader, agent_config
from tests.test_skill_review import document
from videospy.agent import VideoSpyAgent


def test_document_skill_does_not_force_early_stopping(monkeypatch):
    config = agent_config()
    config['tools']['memory_search']['enabled'] = False
    config['tools']['enabled'] = ['frame_inspect']
    config['agent']['max_steps'] = 4
    config['TASK_SKILLS'] = {'general': document('Stop after duplicate evidence when the question is resolved.')}
    responses = [{'role': 'assistant', 'tool_calls': [{'id': str(i), 'function': {
        'name': 'frame_inspect', 'arguments': '{"timestamps":[1],"query":"read the label"}'}}]} for i in range(4)]
    client = Mock()
    client.chat.side_effect = responses + [{'role': 'assistant', 'content': 'A'}]
    monkeypatch.setattr('videospy.agent.ModelClient', Mock(return_value=client))
    config['agent']['include_subtitles_in_question'] = False
    agent = VideoSpyAgent(config, 'video', None, '/tmp', [], video_reader=FakeVideoReader())
    agent.tools = [{'type': 'function', 'function': {'name': 'frame_inspect'}}]
    agent._execute_tool_call = Mock(side_effect=[
        ({'role': 'tool', 'tool_call_id': str(i), 'content': 'The label is visible.' if i == 0 else 'Duplicate `frame_inspect` request.'}, True)
        for i in range(4)])
    result = agent.run('Read the label?')
    assert result.finish_reason == 'reach_max_steps'
    assert client.chat.call_count == 5 and client.chat.call_args.kwargs['tools'] == []
