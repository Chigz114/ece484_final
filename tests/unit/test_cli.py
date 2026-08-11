"""Tests for the unified, dependency-light command router."""

from __future__ import annotations

import contextlib
import io
import unittest

from quadpilot.cli.main import main


class CommandRouterTests(unittest.TestCase):
    def test_top_level_help_lists_primary_workflows(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(["--help"])
        self.assertEqual(result, 0)
        text = output.getvalue()
        self.assertIn("data generate uniform", text)
        self.assertIn("train npe", text)
        self.assertIn("simulate closed-loop", text)
        self.assertIn("hardware preflight", text)

    def test_unknown_command_fails_closed(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            result = main(["unknown"])
        self.assertEqual(result, 2)
        self.assertIn("Unknown command", output.getvalue())

    def test_leaf_help_is_forwarded(self) -> None:
        output = io.StringIO()
        with self.assertRaises(SystemExit) as raised:
            with contextlib.redirect_stdout(output):
                main(["verify", "assets", "--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("asset preflight", output.getvalue())


if __name__ == "__main__":
    unittest.main()
