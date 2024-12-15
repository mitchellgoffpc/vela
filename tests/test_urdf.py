import io
import unittest
from vela.urdf import parse_urdf

class TestURDFParser(unittest.TestCase):

    # Test successful parsing

    def test_single_link_and_joint(self):
        """Test parsing of a minimal correct URDF."""
        xml_content = """<robot name="test_robot">
            <link name="link1">
                <visual>
                    <origin xyz="1 0 0" rpy="0 0 0"/>
                    <geometry>
                        <cylinder length="0.6" radius="0.2"/>
                    </geometry>
                </visual>
            </link>
            <joint name="joint1">
            </joint>
        </robot>"""
        f = io.StringIO(xml_content)
        links, joints = parse_urdf(f)
        self.assertEqual(len(links), 1)
        self.assertEqual(len(joints), 1)
        self.assertEqual(links[0].name, 'link1')
        self.assertIsNotNone(links[0].visual)
        self.assertEqual(links[0].visual.origin.xyz.tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(links[0].visual.origin.rpy.tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(joints[0].name, 'joint1')

    def test_multiple_links_and_joints(self):
        """Test correct parsing of multiple links and joints."""
        xml_content = """<robot name="test_robot">
            <link name="link1">
                <visual>
                    <origin xyz="1 0 0" rpy="0 0 0"/>
                    <geometry>
                        <cylinder length="0.6" radius="0.2"/>
                    </geometry>
                </visual>
            </link>
            <link name="link2">
                <visual>
                    <origin xyz="0 1 0" rpy="0 0 0"/>
                    <geometry>
                        <cylinder length="0.6" radius="0.2"/>
                    </geometry>
                </visual>
            </link>
            <joint name="joint1"/>
            <joint name="joint2"/>
        </robot>"""
        f = io.StringIO(xml_content)
        links, joints = parse_urdf(f)
        self.assertEqual(len(links), 2)
        self.assertEqual(links[0].name, 'link1')
        self.assertEqual(links[1].name, 'link2')
        self.assertEqual(len(joints), 2)
        self.assertEqual(joints[0].name, 'joint1')
        self.assertEqual(joints[1].name, 'joint2')

    def test_missing_link_elements(self):
        """Test that missing 'link' elements results in an empty link list."""
        xml_content = """<robot name="test_robot">
            <!-- No link elements -->
        </robot>"""
        f = io.StringIO(xml_content)
        links, joints = parse_urdf(f)
        self.assertEqual(len(links), 0)
        self.assertEqual(len(joints), 0)

    def test_missing_visual(self):
        """Test parsing when 'visual' is missing but 'collision' is present."""
        xml_content = """<robot name="test_robot">
            <link name="link1">
                <!-- Missing 'visual' element -->
                <collision>
                    <origin xyz="1 0 0" rpy="0 0 0"/>
                    <geometry>
                        <cylinder length="0.6" radius="0.2"/>
                    </geometry>
                </collision>
            </link>
        </robot>"""
        f = io.StringIO(xml_content)
        links, _ = parse_urdf(f)
        self.assertEqual(len(links), 1)
        self.assertIsNone(links[0].visual)
        self.assertIsNotNone(links[0].collision)
        self.assertEqual(links[0].collision.origin.xyz.tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(links[0].collision.origin.rpy.tolist(), [0.0, 0.0, 0.0])

    def test_missing_collision(self):
        """Test parsing when 'collision' is missing but 'visual' is present."""
        xml_content = """<robot name="test_robot">
            <link name="link1">
                <visual>
                    <origin xyz="1 0 0" rpy="0 0 0"/>
                    <geometry>
                        <cylinder length="0.6" radius="0.2"/>
                    </geometry>
                </visual>
                <!-- Missing 'collision' element -->
            </link>
        </robot>"""
        f = io.StringIO(xml_content)
        links, _ = parse_urdf(f)
        self.assertEqual(len(links), 1)
        self.assertIsNotNone(links[0].visual)
        self.assertIsNone(links[0].collision)
        self.assertEqual(links[0].visual.origin.xyz.tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(links[0].visual.origin.rpy.tolist(), [0.0, 0.0, 0.0])

    def test_missing_link_name(self):
        """Test that links without 'name' attribute are named 'anonymous'."""
        xml_content = """<robot name="test_robot">
            <link>
                <visual>
                    <origin xyz="1 0 0" rpy="0 0 0"/>
                    <geometry>
                        <cylinder length="0.6" radius="0.2"/>
                    </geometry>
                </visual>
            </link>
        </robot>"""
        f = io.StringIO(xml_content)
        links, _ = parse_urdf(f)
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0].name, 'anonymous')

    def test_missing_joint_name(self):
        """Test that joints without 'name' attribute are named 'anonymous'."""
        xml_content = """<robot name="test_robot">
            <joint>
            </joint>
        </robot>"""
        f = io.StringIO(xml_content)
        _, joints = parse_urdf(f)
        self.assertEqual(len(joints), 1)
        self.assertEqual(joints[0].name, 'anonymous')


    # Test sanity checks

    def test_missing_origin_element(self):
        """Test that missing 'origin' element raises ValueError."""
        xml_content = """<robot name="test_robot">
            <link name="link1">
                <visual>
                    <!-- Missing origin element -->
                </visual>
            </link>
        </robot>"""
        f = io.StringIO(xml_content)
        with self.assertRaises(ValueError) as context:
            parse_urdf(f)
        self.assertIn("Missing required origin element in link[name='link1']/visual", str(context.exception))

    def test_missing_attributes(self):
        """Test that missing 'xyz' or 'rpy' attributes raise ValueError."""
        xml_content = """<robot name="test_robot">
            <link name="link1">
                <visual>
                    <origin {attributes}/>
                </visual>
            </link>
        </robot>"""
        test_cases = {
            'xyz': {'attributes': 'rpy="0 0 0"'},
            'rpy': {'attributes': 'xyz="1 0 0"'}}

        for attr, formatting in test_cases.items():
            with self.subTest(attribute=attr):
                f = io.StringIO(xml_content.format(**formatting))
                with self.assertRaises(ValueError) as context:
                    parse_urdf(f)
                self.assertIn(f"Missing attribute link[name='link1']/visual/origin[{attr}]", str(context.exception))

    def test_wrong_number_of_values(self):
        """Test that wrong number of values in 'xyz' and 'rpy' raises ValueError."""
        xml_content = """<robot name="test_robot">
            <link name="link1">
                <visual>
                    <origin {attributes} />
                </visual>
            </link>
        </robot>"""
        test_cases = {
            'xyz': {'attributes': 'xyz="1 0" rpy="0 0 0"'},
            'rpy': {'attributes': 'xyz="1 0 0" rpy="0 0 0 0"'}}

        for attr, formatting in test_cases.items():
            with self.subTest(attribute=attr):
                f = io.StringIO(xml_content.format(**formatting))
                with self.assertRaises(ValueError) as context:
                    parse_urdf(f)
                self.assertIn(f"Attribute link[name='link1']/visual/origin[{attr}] must have exactly 3 values", str(context.exception))

    def test_non_numeric_values(self):
        """Test that non-numeric values in 'xyz' and 'rpy' raise ValueError."""
        xml_content = """<robot name="test_robot">
            <link name="link1">
                <visual>
                    <origin {attributes} />
                </visual>
            </link>
        </robot>"""
        test_cases = {
            'xyz': {'attributes': 'xyz="a b c" rpy="0 0 0"'},
            'rpy': {'attributes': 'xyz="1 0 0" rpy="x y z"'}}

        for attr, formatting in test_cases.items():
            with self.subTest(attribute=attr):
                f = io.StringIO(xml_content.format(**formatting))
                with self.assertRaises(ValueError) as context:
                    parse_urdf(f)
                self.assertIn(f"Invalid format for attribute link[name='link1']/visual/origin[{attr}]", str(context.exception))

    def test_missing_geometry(self):
        """Test that missing 'geometry' element raises ValueError."""
        xml_content = """<robot name="test_robot">
            <link name="link1">
                <visual>
                    <origin xyz="1 0 0" rpy="0 0 0"/>
                    <!-- Missing geometry element -->
                </visual>
            </link>
        </robot>"""
        f = io.StringIO(xml_content)
        with self.assertRaises(ValueError) as context:
            parse_urdf(f)
        self.assertIn("Missing required geometry element in link[name='link1']/visual", str(context.exception))


if __name__ == '__main__':
    unittest.main()
