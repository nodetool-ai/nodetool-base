import sys
import unittest
import asyncio
import os
from unittest.mock import patch, MagicMock
from pydantic import BaseModel

class TestShowNotificationSecurity(unittest.TestCase):
    def setUp(self):
        # We need to ensure src is in path
        src_path = os.path.join(os.getcwd(), "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        # Create mocks for dependencies
        # IMPORTANT: We cannot mock 'nodetool' directly if we want to import 'nodetool.nodes.lib.os'
        # because if sys.modules['nodetool'] is a MagicMock, then 'nodetool.nodes' will be an attribute of that mock,
        # not the real package.

        # We only want to mock the missing pieces.

        self.mock_modules = {
            "nodetool.config": MagicMock(),
            "nodetool.config.environment": MagicMock(),
            "nodetool.workflows": MagicMock(),
            "nodetool.workflows.base_node": MagicMock(),
            "nodetool.workflows.processing_context": MagicMock(),
            "nodetool.metadata": MagicMock(),
            "nodetool.metadata.types": MagicMock(),
        }

        # Setup specific mocks
        mock_base_node_mod = MagicMock()
        mock_base_node_mod.BaseNode = BaseModel
        self.mock_modules["nodetool.workflows.base_node"] = mock_base_node_mod

        # Mock Environment.is_production to default to False for importing
        mock_env = MagicMock()
        mock_env.Environment.is_production.return_value = False
        self.mock_modules["nodetool.config.environment"] = mock_env

        # Patch sys.modules
        self.patcher = patch.dict(sys.modules, self.mock_modules)
        self.patcher.start()

        # We need to make sure nodetool.nodes.lib.os is loaded fresh or reloaded
        # to pick up the mocks if it was already loaded (unlikely in fresh run but good practice)
        if "nodetool.nodes.lib.os" in sys.modules:
             del sys.modules["nodetool.nodes.lib.os"]

    def tearDown(self):
        self.patcher.stop()

    @patch("subprocess.run")
    @patch("nodetool.nodes.lib.os.os.name", "posix")
    @patch("nodetool.nodes.lib.os.os.uname")
    def test_command_injection_prevention(self, mock_uname, mock_subprocess):
        """Test that ShowNotification on macOS uses arguments instead of interpolation."""

        # Import the class under test
        # This import must happen AFTER patching sys.modules
        from nodetool.nodes.lib.os import ShowNotification

        # Setup mock for macOS
        mock_uname_struct = MagicMock()
        mock_uname_struct.sysname = "Darwin"
        mock_uname.return_value = mock_uname_struct

        # Payloads that would cause issues with interpolation
        dangerous_inputs = [
            '"; say "pwned"; --',
            "foo'bar",
            'foo"bar',
            '$(rm -rf /)',
            '`touch /tmp/hacked`'
        ]

        for payload in dangerous_inputs:
            with self.subTest(payload=payload):
                node = ShowNotification(title=payload, message="safe message")

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                context = MagicMock()

                loop.run_until_complete(node.process(context))

                args, _ = mock_subprocess.call_args
                cmd_list = args[0]

                # Verify structure
                self.assertEqual(cmd_list[0], "osascript")
                self.assertEqual(cmd_list[1], "-e")

                # The script (3rd arg) should be static and use argv
                script = cmd_list[2]
                self.assertIn("on run argv", script)
                self.assertIn("item 1 of argv", script)
                self.assertIn("item 2 of argv", script)

                # Verify payload is NOT in the script
                self.assertNotIn(payload, script)

                # Verify -- separator is present
                self.assertEqual(cmd_list[3], "--")

                # Verify payload IS passed as an argument
                self.assertIn(payload, cmd_list)

                loop.close()

if __name__ == "__main__":
    unittest.main()
