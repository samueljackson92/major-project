import command
import nose.tools
import os.path

from click.testing import CliRunner
from ..test_utils import get_file_path


def test_prints_version():
    runner = CliRunner()
    result = runner.invoke(command.cli, ['--version'])
    nose.tools.assert_equal(result.exit_code, 0)
    nose.tools.assert_true('mia, version' in result.output)
