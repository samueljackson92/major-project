import command
import nose.tools
import os.path

from click.testing import CliRunner
from test_utils import get_file_path


def test_prints_version():
    runner = CliRunner()
    result = runner.invoke(command.cli, ['--version'])
    nose.tools.assert_equal(result.exit_code, 0)
    nose.tools.assert_true('mia, version' in result.output)


def test_analysis():
    runner = CliRunner()
    path = get_file_path("blob_detection.csv")
    columns = ['avg_radius', 'std_radius', 'blob_count']
    result = runner.invoke(command.tSNE, [path] + columns)

    nose.tools.assert_equal(result.exit_code, 0)


def test_analysis_saves_to_file():
    runner = CliRunner()
    path = get_file_path("blob_detection.csv")
    with open(path, 'r') as f:
        contents = f.read()

    with runner.isolated_filesystem():
        with open('blob_detection.csv', 'w') as f:
            f.write(contents)

        args = ['blob_detection.csv', '--output-file=output.csv']
        result = runner.invoke(command.tSNE, args)
        nose.tools.assert_equal(result.exit_code, 0)
        nose.tools.assert_true(os.path.isfile('output.csv'))
