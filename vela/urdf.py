import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# Common dataclasses

@dataclass
class Origin:
    xyz: list[float]
    rpy: list[float]


# Link dataclasses

@dataclass
class Box:
    size: list[float]

@dataclass
class Cylinder:
    length: float
    radius: float

@dataclass
class Sphere:
    radius: float

@dataclass
class Mesh:
    filename: str

@dataclass
class Model:
    origin: Origin
    geometry: Cylinder | Sphere | Box | Mesh

@dataclass
class Link:
    name: str
    visual: list[Model]
    collision: list[Model]


# Joint dataclasses

@dataclass
class Axis:
    xyz: list[float]

@dataclass
class Limit:
    lower: float
    upper: float
    effort: float
    velocity: float

@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: Origin
    axis: Optional[Axis]
    limit: Optional[Limit]


# Common parsers

def parse_float(text: Optional[str], attr_path: str) -> float:
    if text is None:
        raise ValueError(f"Missing required attribute {attr_path}")
    try:
        return float(text)
    except ValueError as e:
        raise ValueError(f"Invalid format for attribute {attr_path}: {text}") from e

def parse_array(text: Optional[str], attr_path: str) -> list[float]:
    if text is None:
        raise ValueError(f"Missing required attribute {attr_path}")
    try:
        values = [float(x) for x in text.split()]
    except ValueError as e:
        raise ValueError(f"Invalid format for attribute {attr_path}: {text}") from e
    if len(values) != 3:
        raise ValueError(f"Attribute {attr_path} must have exactly 3 values")
    return values

def parse_origin(elem: ET.Element, tag_path: str) -> Origin:
    xyz, rpy = [0] * 3, [0] * 3
    origin_elems = elem.findall('origin')
    if len(origin_elems) > 1:
        raise ValueError(f"Element {tag_path} must contain at most one 'origin' tag")
    elif len(origin_elems) == 1:
        if (xyz_attr := origin_elems[0].get('xyz')) is not None:
            xyz = parse_array(xyz_attr, f'{tag_path}/origin/@xyz')
        if (rpy_attr := origin_elems[0].get('rpy')) is not None:
            rpy = parse_array(rpy_attr, f'{tag_path}/origin/@rpy')
    return Origin(xyz=xyz, rpy=rpy)


# Link parsers

def parse_box(elem: ET.Element, tag_path: str) -> Box:
    size = parse_array(elem.get('size'), f'{tag_path}/@size')
    return Box(size=size)

def parse_cylinder(elem: ET.Element, tag_path: str) -> Cylinder:
    length = parse_float(elem.get('length'), f'{tag_path}/@length')
    radius = parse_float(elem.get('radius'), f'{tag_path}/@radius')
    return Cylinder(length=length, radius=radius)

def parse_sphere(elem: ET.Element, tag_path: str) -> Sphere:
    radius = parse_float(elem.get('radius'), f'{tag_path}/@radius')
    return Sphere(radius=radius)

def parse_mesh(elem: ET.Element, tag_path: str) -> Mesh:
    if (filename := elem.get('filename')) is None:
        raise ValueError(f"Missing required attribute {tag_path}/@filename")
    return Mesh(filename=filename)

def parse_geometry(elem: ET.Element, tag_path: str) -> Box | Cylinder | Sphere | Mesh:
    geometry_elems = elem.findall('geometry')
    if len(geometry_elems) != 1:
        raise ValueError(f"Element {tag_path} must contain exactly one 'geometry' tag")

    children = list(geometry_elems[0])
    if len(children) != 1:
        raise ValueError(f"Element {tag_path}/geometry must contain exactly one child element")

    child = children[0]
    child_path = f'{tag_path}/geometry/{child.tag}'
    match child.tag:
        case 'box': return parse_box(child, child_path)
        case 'cylinder': return parse_cylinder(child, child_path)
        case 'sphere': return parse_sphere(child, child_path)
        case 'mesh': return parse_mesh(child, child_path)
        case _: raise ValueError(f"Invalid geometry type '{child.tag}' in {tag_path}/geometry")

def parse_model(elems: list[ET.Element], tag_name: str) -> list[Model]:
    return [Model(origin=parse_origin(elem, f'{tag_name}[{i}]'),
                  geometry=parse_geometry(elem, f'{tag_name}[{i}]'))
            for i, elem in enumerate(elems)]

def parse_link(elem: ET.Element) -> Link:
    link_name = elem.get('name')
    if link_name is None:
        raise ValueError("Missing required attribute 'name' in link element")
    visual = parse_model(elem.findall('visual'), f"/link[@name='{link_name}']/visual")
    collision = parse_model(elem.findall('collision'), f"/link[@name='{link_name}']/collision")
    return Link(name=link_name, visual=visual, collision=collision)


# Joint parsers

def parse_axis(elem: ET.Element, tag_path: str) -> Optional[Axis]:
    axis_elems = elem.findall('axis')
    if len(axis_elems) > 1:
        raise ValueError(f"Element {tag_path} must contain at most one 'axis' tag")
    elif len(axis_elems) == 1:
        xyz = parse_array(axis_elems[0].get('xyz'), f'{tag_path}/axis/@xyz')
        return Axis(xyz=xyz)
    else:
        return None

def parse_limit(elem: ET.Element, tag_path: str) -> Optional[Limit]:
    limit_elems = elem.findall('limit')
    if len(limit_elems) > 1:
        raise ValueError(f"Element {tag_path} must contain at most one 'limit' tag")
    elif len(limit_elems) == 1:
        lower, upper = 0, 0
        if (lower_attr := limit_elems[0].get('lower')) is not None:
            lower = parse_float(lower_attr, f'{tag_path}/limit/@lower')
        if (upper_attr := limit_elems[0].get('upper')) is not None:
            upper = parse_float(upper_attr, f'{tag_path}/limit/@upper')
        effort = parse_float(limit_elems[0].get('effort'), f'{tag_path}/limit/@effort')
        velocity = parse_float(limit_elems[0].get('velocity'), f'{tag_path}/limit/@velocity')
        return Limit(lower=lower, upper=upper, effort=effort, velocity=velocity)
    else:
        return None

def parse_joint(elem: ET.Element) -> Joint:
    if (joint_name := elem.get('name')) is None:
        raise ValueError("Missing required attribute 'name' in joint element")
    if (joint_type := elem.get('type')) is None:
        raise ValueError(f"Missing required attribute 'type' in joint element '{joint_name}'")
    elif joint_type not in ('revolute', 'continuous', 'prismatic', 'fixed', 'floating', 'planar'):
        raise ValueError(f"Invalid joint type '{joint_type}' in joint element '{joint_name}'")

    parent_elems = elem.findall('parent')
    if len(parent_elems) != 1:
        raise ValueError(f"Element /joint[@name='{joint_name}'] must contain exactly one 'parent' tag")
    if (parent := parent_elems[0].get('link')) is None:
        raise ValueError(f"Missing required attribute 'link' in parent element of joint '{joint_name}'")

    child_elems = elem.findall('child')
    if len(child_elems) != 1:
        raise ValueError(f"Element /joint[@name='{joint_name}'] must contain exactly one 'child' tag")
    if (child := child_elems[0].get('link')) is None:
        raise ValueError(f"Missing required attribute 'link' in child element of joint '{joint_name}'")

    origin = parse_origin(elem, f"/joint[@name='{joint_name}']")
    axis = parse_axis(elem, f"/joint[@name='{joint_name}']")
    limit = parse_limit(elem, f"/joint[@name='{joint_name}']")
    if joint_type in ('revolute', 'continuous', 'prismatic', 'planar') and axis is None:
        raise ValueError(f"Joint '{joint_name}' of type '{joint_type}' must contain an 'axis' tag")
    if joint_type in ('revolute', 'prismatic') and limit is None:
        raise ValueError(f"Joint '{joint_name}' of type '{joint_type}' must contain a 'limit' tag")
    return Joint(name=joint_name, type=joint_type, parent=parent, child=child, origin=origin, axis=axis, limit=limit)


# URDF parsers

def parse_urdf(path: Path) -> tuple[list[Link], list[Joint]]:
    tree = ET.parse(path)
    root = tree.getroot()
    links = [parse_link(elem) for elem in root.findall('link')]
    joints = [parse_joint(elem) for elem in root.findall('joint')]
    return links, joints
