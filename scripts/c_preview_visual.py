"""Use the continuous HY-Motion surface for collaborator previews."""
from pathlib import Path
import xml.etree.ElementTree as ET
from intent_policy.sim.scene_builder_v2 import add_continuous_human_skin


def connected_human(scene, xml):
    scene['human']['continuous_skin'] = True
    root = ET.fromstring(xml)
    add_continuous_human_skin(root, Path(__file__).resolve().parents[1] / 'assets/human')
    return ET.tostring(root, encoding='unicode')
