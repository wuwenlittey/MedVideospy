"""Read inspected frame positions from program-generated tool messages."""
import re


def frame_timestamps(message):
    """Read inspected positions from the tool header, not the model's description."""
    if message.get('role') != 'tool' or message.get('name') != 'frame_inspect':
        return []
    match = re.match(r'Frame inspection at ([0-9.,s ]+) \(requested [0-9.,s ]+\):',
                     str(message.get('content', '')))
    return sorted({float(t) for t in re.findall(r'(\d+(?:\.\d+)?)s', match[1])}) if match else []
