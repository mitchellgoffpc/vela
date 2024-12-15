import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

@dataclass
class Origin:
    xyz: np.ndarray
    rpy: np.ndarray

@dataclass
class Geometry:
    tag: str

@dataclass
class Model:
    origin: Origin
    geometry: Geometry

@dataclass
class Link:
    name: str
    visual: Optional[Model] = None
    collision: Optional[Model] = None

@dataclass
class Joint:
    name: str


def parse_array(text: Optional[str], attr_path: str) -> np.ndarray:
    if text is None:
        raise ValueError(f"Missing attribute {attr_path}")
    try:
        values = [float(x) for x in text.split()]
    except ValueError as e:
        raise ValueError(f"Invalid format for attribute {attr_path}: {text}") from e
    if len(values) != 3:
        raise ValueError(f"Attribute {attr_path} must have exactly 3 values")
    return np.array(values)

def parse_origin(elem: ET.Element, parent_path: str) -> Origin:
    origin_elem = elem.find('origin')
    if origin_elem is None:
        raise ValueError(f"Missing required origin element in {parent_path}")
    tag_path = f"{parent_path}/origin"
    return Origin(
        xyz=parse_array(origin_elem.get('xyz'), f'{tag_path}[xyz]'),
        rpy=parse_array(origin_elem.get('rpy'), f'{tag_path}[rpy]'))

def parse_geometry(elem: ET.Element, parent_path: str) -> Geometry:
    geometry_elem = elem.find('geometry')
    if geometry_elem is None:
        raise ValueError(f"Missing required geometry element in {parent_path}")

    children = list(geometry_elem)
    if len(children) != 1:
        raise ValueError(f"Geometry element in {parent_path} must have exactly one child element")

    child = children[0]
    if child.tag not in ('cylinder', 'box', 'sphere', 'mesh'):
        raise ValueError(f"Invalid geometry type {child.tag} in {parent_path}")

    return Geometry(tag=child.tag)

def parse_model(elem: Optional[ET.Element], tag_name: str) -> Optional[Model]:
    if elem is None:
        return None
    return Model(
        origin=parse_origin(elem, f"{tag_name}"),
        geometry=parse_geometry(elem, f"{tag_name}")
    )

def parse_urdf(path: Path) -> tuple[list[Link], list[Joint]]:
    tree = ET.parse(path)
    root = tree.getroot()

    links = []
    for link in root.findall('link'):
        link_name = link.get('name', 'anonymous')
        visual = parse_model(link.find('visual'), f"link[name='{link_name}']/visual")
        collision = parse_model(link.find('collision'), f"link[name='{link_name}']/collision")
        links.append(Link(name=link_name, visual=visual, collision=collision))

    joints = []
    for joint in root.findall('joint'):
        joint_name = joint.get('name', 'anonymous')
        joints.append(Joint(name=joint_name))

    return links, joints
