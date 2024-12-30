import io
import unittest
from vela.helpers.urdf import parse_urdf, Origin, Box, Cylinder, Sphere, Mesh, Axis, Limit

class TestWellFormedLinks(unittest.TestCase):
    def test_basic_link(self):
        """Test that links with basic attributes are parsed correctly."""
        xml_content = """<robot name="test_robot">
            <link name="link1">
                <visual>
                    <origin xyz="1.5 0 0.0" rpy="0.0 0 0.2"/>
                    <geometry>
                        <cylinder length="0.6" radius="0.2"/>
                    </geometry>
                </visual>
            </link>
        </robot>"""
        f = io.StringIO(xml_content)
        links, _ = parse_urdf(f)
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0].name, 'link1')
        self.assertEqual(len(links[0].visual), 1)
        self.assertEqual(links[0].visual[0].origin, Origin(xyz=[1.5, 0.0, 0.0], rpy=[0.0, 0.0, 0.2]))

    def test_link_no_visual_or_collision(self):
        """Test that a link with no visual or collision elements is parsed correctly."""
        xml_content = """<robot name="test_robot">
            <link name="link1"></link>
        </robot>"""
        f = io.StringIO(xml_content)
        links, _ = parse_urdf(f)
        self.assertEqual(links[0].visual, [])
        self.assertEqual(links[0].collision, [])

    def test_missing_origin_element(self):
        """Test that missing 'origin' element raises ValueError."""
        xml_content = """<robot name="test_robot">
            <link name="link1">
                <visual>
                    <geometry>
                        <cylinder length="0.6" radius="0.2"/>
                    </geometry>
                </visual>
            </link>
        </robot>"""
        f = io.StringIO(xml_content)
        links, _ = parse_urdf(f)
        self.assertEqual(links[0].visual[0].origin, Origin(xyz=[0.0, 0.0, 0.0], rpy=[0.0, 0.0, 0.0]))

    def test_missing_origin_attributes(self):
        """Test that missing 'xyz' or 'rpy' attributes default to zeros."""
        xml_content = """<robot name="test_robot">
            <link name="link1">
                <visual>
                    <origin {attributes}/>
                    <geometry>
                        <cylinder length="0.6" radius="0.2"/>
                    </geometry>
                </visual>
            </link>
        </robot>"""
        test_cases = {
            'xyz': {'attributes': 'rpy="0 0 0"', 'expected_xyz': [0.0, 0.0, 0.0], 'expected_rpy': [0.0, 0.0, 0.0]},
            'rpy': {'attributes': 'xyz="1 0 0"', 'expected_xyz': [1.0, 0.0, 0.0], 'expected_rpy': [0.0, 0.0, 0.0]}}

        for attr, formatting in test_cases.items():
            with self.subTest(attribute=attr):
                f = io.StringIO(xml_content.format(**formatting))
                links, _ = parse_urdf(f)
                self.assertEqual(links[0].visual[0].origin.xyz, formatting['expected_xyz'])
                self.assertEqual(links[0].visual[0].origin.rpy, formatting['expected_rpy'])

    def test_geometry_types(self):
        """Test that different geometry types are parsed correctly."""
        xml_content = """<robot name="test_robot">
            <link name="link1">
                <visual>
                    <origin xyz="1 0 0" rpy="0 0 0"/>
                    <geometry>
                        {geometry_element}
                    </geometry>
                </visual>
            </link>
        </robot>"""
        test_cases = {
            'box': ('<box size="1 1 1"/>', Box(size=[1, 1, 1])),
            'cylinder': ('<cylinder length="0.6" radius="0.2"/>', Cylinder(length=0.6, radius=0.2)),
            'sphere': ('<sphere radius="0.5"/>', Sphere(radius=0.5)),
            'mesh': ('<mesh filename="mesh.stl" scale="1 1 1"/>', Mesh(filename='mesh.stl', scale=[1.] * 3))}

        for geometry_type, (geometry_element, expected_output) in test_cases.items():
            with self.subTest(geometry=geometry_type):
                f = io.StringIO(xml_content.format(geometry_element=geometry_element))
                links, _ = parse_urdf(f)
                self.assertEqual(links[0].visual[0].geometry, expected_output)

    def test_mesh_scale(self):
        """Test that the 'scale' parameter on the <mesh> tag is parsed correctly."""
        xml_content = """<robot name="test_robot">
            <link name="link1">
                <visual>
                    <origin xyz="1 0 0" rpy="0 0 0"/>
                    <geometry>
                        <mesh filename="mesh.stl" {scale_attr}/>
                    </geometry>
                </visual>
            </link>
        </robot>"""
        test_cases = [
            ('', [1.0, 1.0, 1.0]),
            ('scale="2 3 4"', [2.0, 3.0, 4.0]),
            ('scale="0.5 0.5 0.5"', [0.5, 0.5, 0.5])]

        for scale_attr, expected_scale in test_cases:
            with self.subTest(scale=scale_attr):
                f = io.StringIO(xml_content.format(scale_attr=scale_attr))
                links, _ = parse_urdf(f)
                self.assertIsInstance(links[0].visual[0].geometry, Mesh)
                self.assertEqual(links[0].visual[0].geometry.scale, expected_scale)


class TestWellFormedJoints(unittest.TestCase):
    def test_basic_joint(self):
        """Test that joints with basic attributes are parsed correctly."""
        xml_content = """<robot name="test_robot">
            <link name="link1"/>
            <link name="link2"/>
            <joint name="joint1" type="revolute">
                <parent link="link1"/>
                <child link="link2"/>
                <origin xyz="1 0 0" rpy="0 0 1"/>
                <axis xyz="0 0 1"/>
                <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
            </joint>
        </robot>"""

        f = io.StringIO(xml_content)
        _, joints = parse_urdf(f)
        self.assertEqual(len(joints), 1)
        self.assertEqual(joints[0].name, 'joint1')
        self.assertEqual(joints[0].type, 'revolute')
        self.assertEqual(joints[0].parent, 'link1')
        self.assertEqual(joints[0].child, 'link2')
        self.assertEqual(joints[0].origin, Origin(xyz=[1.0, 0.0, 0.0], rpy=[0.0, 0.0, 1.0]))
        self.assertEqual(joints[0].axis, Axis(xyz=[0.0, 0.0, 1.0]))
        self.assertEqual(joints[0].limit, Limit(lower=-3.14, upper=3.14, effort=1.0, velocity=1.0))

    def test_all_joint_types(self):
        """Test that all valid joint types are parsed correctly."""
        xml_content = """<robot name="test_robot">
            <link name="link1"/>
            <link name="link2"/>
            <joint name="joint1" type="{joint_type}">
                <parent link="link1"/>
                <child link="link2"/>
                <axis xyz="0 0 1"/>
                <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
            </joint>
        </robot>"""

        for joint_type in ['revolute', 'continuous', 'prismatic', 'fixed', 'floating', 'planar']:
            with self.subTest(type=joint_type):
                f = io.StringIO(xml_content.format(joint_type=joint_type))
                _, joints = parse_urdf(f)
                self.assertEqual(joints[0].type, joint_type)

    def test_missing_origin_element(self):
        """Test that missing 'origin' element raises ValueError."""
        xml_content = """<robot name="test_robot">
            <link name="link1"/>
            <link name="link2"/>
            <joint name="joint1" type="fixed">
                <parent link="link1"/>
                <child link="link2"/>
            </joint>
        </robot>"""
        f = io.StringIO(xml_content)
        _, joints = parse_urdf(f)
        self.assertEqual(joints[0].origin, Origin(xyz=[0.0, 0.0, 0.0], rpy=[0.0, 0.0, 0.0]))

    def test_missing_axis_element(self):
        """Test that joints without 'axis' element raise ValueError."""
        xml_content = """<robot name="test_robot">
            <joint name="joint1" type="fixed">
                <parent link="link1"/>
                <child link="link2"/>
            </joint>
        </robot>"""
        f = io.StringIO(xml_content)
        _, joints = parse_urdf(f)
        self.assertEqual(joints[0].axis, Axis(xyz=[1.0, 0.0, 0.0]))

    def test_missing_limit_bounds(self):
        """Test that missing 'lower' or 'upper' attributes default to zeros."""
        xml_content = """<robot name="test_robot">
            <link name="link1"/>
            <link name="link2"/>
            <joint name="joint1" type="revolute">
                <parent link="link1"/>
                <child link="link2"/>
                <axis xyz="0 0 1"/>
                <limit effort="1" velocity="1"/>
            </joint>
        </robot>"""
        f = io.StringIO(xml_content)
        _, joints = parse_urdf(f)
        self.assertEqual(joints[0].limit, Limit(lower=0.0, upper=0.0, effort=1.0, velocity=1.0))


class TestMalformedLinks(unittest.TestCase):
    def test_missing_link_name(self):
        """Test that links without 'name' attribute raise ValueError."""
        xml_content = """<robot name="test_robot">
            <link>
            </link>
        </robot>"""
        f = io.StringIO(xml_content)
        with self.assertRaises(ValueError) as context:
            parse_urdf(f)
        self.assertEqual("Missing required attribute 'name' in link element", str(context.exception))

    def test_multiple_origin_elements(self):
        """Test that multiple 'origin' elements raise ValueError."""
        xml_content = """<robot name="test_robot">
            <link name="link1">
                <visual>
                    <origin xyz="1 0 0" rpy="0 0 0"/>
                    <origin xyz="0 1 0" rpy="0 0 0"/>
                </visual>
            </link>
        </robot>"""
        f = io.StringIO(xml_content)
        with self.assertRaises(ValueError) as context:
            parse_urdf(f)
        self.assertEqual("Element /link[@name='link1']/visual[0] must contain at most one 'origin' tag", str(context.exception))

    def test_malformed_origin_attributes(self):
        """Test that malformed 'xyz' and 'rpy' attributes in origin element raise ValueError."""
        xml_content = """<robot name="test_robot">
            <link name="link1">
                <visual>
                    <origin {attributes}/>
                </visual>
            </link>
        </robot>"""
        test_cases = {
            'xyz="1 0" rpy="0 0 0"': "Attribute /link[@name='link1']/visual[0]/origin/@xyz must have exactly 3 values",
            'xyz="1 0 0" rpy="0 0 0 0"': "Attribute /link[@name='link1']/visual[0]/origin/@rpy must have exactly 3 values",
            'xyz="a b c" rpy="0 0 0"': "Invalid format for attribute /link[@name='link1']/visual[0]/origin/@xyz: a b c",
            'xyz="1 0 0" rpy="x y z"': "Invalid format for attribute /link[@name='link1']/visual[0]/origin/@rpy: x y z"}

        for attributes, error_msg in test_cases.items():
            with self.subTest(attributes=attributes):
                f = io.StringIO(xml_content.format(attributes=attributes))
                with self.assertRaises(ValueError) as context:
                    parse_urdf(f)
                self.assertEqual(error_msg, str(context.exception))

    def test_wrong_geometry_element_count(self):
        """Test that missing or multiple geometry elements raise ValueError."""
        xml_content = """<robot name="test_robot">
            <link name="link1">
                <visual>
                    <origin xyz="1 0 0" rpy="0 0 0"/>
                    {geometry_elements}
                </visual>
            </link>
        </robot>"""
        test_cases = {
            'missing': '',
            'multiple': '<geometry><cylinder length="0.6" radius="0.2"/></geometry><geometry><cylinder length="0.6" radius="0.2"/></geometry>'}

        for case, geometry_elements in test_cases.items():
            with self.subTest(case=case):
                f = io.StringIO(xml_content.format(geometry_elements=geometry_elements))
                with self.assertRaises(ValueError) as context:
                    parse_urdf(f)
                self.assertEqual("Element /link[@name='link1']/visual[0] must contain exactly one 'geometry' tag", str(context.exception))

    def test_wrong_geometry_child_count(self):
        """Test that missing or multiple geometry children raise ValueError."""
        xml_content = """<robot name="test_robot">
            <link name="link1">
                <visual>
                    <origin xyz="1 0 0" rpy="0 0 0"/>
                    <geometry>
                        {geometry_children}
                    </geometry>
                </visual>
            </link>
        </robot>"""
        test_cases = {
            'missing': '',
            'multiple': '<cylinder length="0.6" radius="0.2"/><box size="1 1 1"/>'}

        for case, geometry_children in test_cases.items():
            with self.subTest(case=case):
                f = io.StringIO(xml_content.format(geometry_children=geometry_children))
                with self.assertRaises(ValueError) as context:
                    parse_urdf(f)
                self.assertEqual("Element /link[@name='link1']/visual[0]/geometry must contain exactly one child element", str(context.exception))

    def test_unknown_geometry_type(self):
        """Test that unknown geometry type raises ValueError."""
        xml_content = """<robot name="test_robot">
            <link name="link1">
                <visual>
                    <origin xyz="1 0 0" rpy="0 0 0"/>
                    <geometry>
                        <unknown/>
                    </geometry>
                </visual>
            </link>
        </robot>"""
        f = io.StringIO(xml_content)
        with self.assertRaises(ValueError) as context:
            parse_urdf(f)
        self.assertEqual("Invalid geometry type 'unknown' in /link[@name='link1']/visual[0]/geometry", str(context.exception))

    def test_missing_geometry_attributes(self):
        """Test that missing attributes in geometry elements raise ValueError."""
        xml_content = """<robot name="test_robot">
            <link name="link1">
                <visual>
                    <origin xyz="1 0 0" rpy="0 0 0"/>
                    <geometry>
                        {geometry_element}
                    </geometry>
                </visual>
            </link>
        </robot>"""
        test_cases = {
            'box': [('<box/>', 'size')],
            'cylinder': [('<cylinder radius="0.6"/>', 'length'), ('<cylinder length="0.6"/>', 'radius')],
            'sphere': [('<sphere/>', 'radius')],
            'mesh': [('<mesh/>', 'filename')]}

        for geometry_type, test_subcases in test_cases.items():
            with self.subTest(geometry=geometry_type):
                for geometry_element, missing_attr in test_subcases:
                    f = io.StringIO(xml_content.format(geometry_element=geometry_element))
                    with self.assertRaises(ValueError) as context:
                        parse_urdf(f)
                    self.assertEqual(f"Missing required attribute /link[@name='link1']/visual[0]/geometry/{geometry_type}/@{missing_attr}", str(context.exception))

    def test_malformed_geometry_elements(self):
        """Test that malformed values in geometry elements raise ValueError."""
        xml_content = """<robot name="test_robot">
            <link name="link1">
                <visual>
                    <origin xyz="1 0 0" rpy="0 0 0"/>
                    <geometry>
                        {geometry_element}
                    </geometry>
                </visual>
            </link>
        </robot>"""
        test_cases = {
            'box': [('<box size="xyz"/>', 'size')],
            'cylinder': [('<cylinder length="xyz" radius="0.2"/>', 'length'), ('<cylinder length="0.6" radius="xyz"/>', 'radius')],
            'sphere': [('<sphere radius="xyz"/>', 'radius')]}

        for geometry_type, test_subcases in test_cases.items():
            with self.subTest(geometry=geometry_type):
                for geometry_element, malformed_attr in test_subcases:
                    f = io.StringIO(xml_content.format(geometry_element=geometry_element))
                    with self.assertRaises(ValueError) as context:
                        parse_urdf(f)
                    self.assertIn(f"Invalid format for attribute /link[@name='link1']/visual[0]/geometry/{geometry_type}/@{malformed_attr}", str(context.exception))


class TestMalformedJoints(unittest.TestCase):
    def test_malformed_joint_element(self):
        """Test that joints with malformed elements raise ValueError."""
        xml_content = '<robot name="test_robot">{joint_element}</robot>'
        test_cases = {
            'missing_name': ('<joint type="revolute"/>', "Missing required attribute 'name' in joint element"),
            'missing_type': ('<joint name="joint1"/>', "Missing required attribute 'type' in joint element 'joint1'"),
            'unknown_type': ('<joint name="joint1" type="unknown"/>', "Invalid joint type 'unknown' in joint element 'joint1'")}

        for case, (joint_element, error_msg) in test_cases.items():
            with self.subTest(case=case):
                f = io.StringIO(xml_content.format(joint_element=joint_element))
                with self.assertRaises(ValueError) as context:
                    parse_urdf(f)
                self.assertEqual(error_msg, str(context.exception))

    def test_wrong_parent_or_child_element_count(self):
        """Test that joints without 'parent' or 'child' element or with multiple 'parent' or 'child' elements raise ValueError."""
        xml_content = """<robot name="test_robot">
            <joint name="joint1" type="revolute">
                {elements}
            </joint>
        </robot>"""
        test_cases = {
            'missing_parent': ('parent', '<child link="link1"/>'),
            'missing_child': ('child', '<parent link="link1"/>'),
            'multiple_parent': ('parent', '<parent link="link1"/><parent link="link2"/><child link="link1"/>'),
            'multiple_child': ('child', '<parent link="link1"/><child link="link1"/><child link="link2"/>')}

        for case, (tag_name, elements) in test_cases.items():
            with self.subTest(case=case):
                f = io.StringIO(xml_content.format(elements=elements))
                with self.assertRaises(ValueError) as context:
                    parse_urdf(f)
                self.assertEqual(f"Element /joint[@name='joint1'] must contain exactly one '{tag_name}' tag", str(context.exception))

    def test_missing_link_attribute(self):
        """Test that joints without 'link' attribute in 'parent' or 'child' element raise ValueError."""
        xml_content = """<robot name="test_robot">
            <joint name="joint1" type="revolute">
                {elements}
            </joint>
        </robot>"""
        test_cases = {
            'parent': '<parent/><child link="link1" />',
            'child': '<child/><parent link="link1" />'}

        for missing_tag, elements in test_cases.items():
            with self.subTest(case=missing_tag):
                f = io.StringIO(xml_content.format(elements=elements))
                with self.assertRaises(ValueError) as context:
                    parse_urdf(f)
                self.assertEqual(f"Missing required attribute 'link' in {missing_tag} element of joint 'joint1'", str(context.exception))

    def test_missing_limit_element(self):
        """Test that joints without 'limit' element raise ValueError."""
        xml_content = """<robot name="test_robot">
            <joint name="joint1" type="{joint_type}">
                <parent link="link1"/>
                <child link="link2"/>
                <axis xyz="0 0 1"/>
            </joint>
        </robot>"""

        for joint_type in ('fixed', 'floating', 'continuous', 'planar'):
            with self.subTest(joint_type=joint_type):
                f = io.StringIO(xml_content.format(joint_type=joint_type))
                _, joints = parse_urdf(f)
                self.assertEqual(joints[0].limit, None)

        for joint_type in ('prismatic', 'revolute'):
            with self.subTest(joint_type=joint_type):
                f = io.StringIO(xml_content.format(joint_type=joint_type))
                with self.assertRaises(ValueError) as context:
                    parse_urdf(f)
                self.assertEqual(f"Joint 'joint1' of type '{joint_type}' must contain a 'limit' tag", str(context.exception))

    def test_multiple_axis_and_limit_elements(self):
        """Test that joints with multiple 'axis' or 'limit' elements raise ValueError."""
        xml_content = """<robot name="test_robot">
            <joint name="joint1" type="revolute">
                <parent link="link1"/>
                <child link="link2"/>
                <axis xyz="0 0 1"/>
                <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
                {elements}
            </joint>
        </robot>"""
        test_cases = {
            'multiple_axis': ('axis', '<axis xyz="0 0 1"/>'),
            'multiple_limit': ('limit', '<limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>')}

        for case, (tag_name, elements) in test_cases.items():
            with self.subTest(case=case):
                f = io.StringIO(xml_content.format(elements=elements))
                with self.assertRaises(ValueError) as context:
                    parse_urdf(f)
                self.assertEqual(f"Element /joint[@name='joint1'] must contain at most one '{tag_name}' tag", str(context.exception))

    def test_malformed_mesh_scale(self):
        """Test that malformed 'scale' attribute in mesh element raises ValueError."""
        xml_content = """<robot name="test_robot">
            <link name="link1">
                <visual>
                    <origin xyz="1 0 0" rpy="0 0 0"/>
                    <geometry>
                        <mesh filename="mesh.stl" scale="{scale_value}"/>
                    </geometry>
                </visual>
            </link>
        </robot>"""
        test_cases = {
            "1 2": "Attribute {attribute} must have exactly 3 values",
            "1 2 3 4": "Attribute {attribute} must have exactly 3 values",
            "a b c": "Invalid format for attribute {attribute}: {value}",
            "1.0 2.0 abc": "Invalid format for attribute {attribute}: {value}"}

        for scale_value, error_msg in test_cases.items():
            with self.subTest(scale=scale_value):
                f = io.StringIO(xml_content.format(scale_value=scale_value))
                with self.assertRaises(ValueError) as context:
                    parse_urdf(f)
                self.assertEqual(error_msg.format(attribute="/link[@name='link1']/visual[0]/geometry/mesh/@scale", value=scale_value), str(context.exception))


if __name__ == '__main__':
    unittest.main()
