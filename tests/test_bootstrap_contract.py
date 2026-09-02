from __future__ import annotations

import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
RESEARCH_SCRIPTS = (
    "extract_isic_chain.sh",
    "finish_utilisation.sh",
    "probe_isic_chain.sh",
    "run_behaviour.sh",
    "run_chain.sh",
    "supervise_cad.sh",
    "supervise_elicit.sh",
    "supervise_plant.sh",
    "supervise.sh",
)


class BootstrapContractTests(unittest.TestCase):
    def test_python_sources_compile(self) -> None:
        for source in sorted((ROOT / "src").glob("*.py")):
            with self.subTest(source=source.name):
                try:
                    compile(source.read_text(encoding="utf-8"), str(source), "exec")
                except SyntaxError as error:
                    self.fail(f"{source.name} does not compile: {error}")

    def test_crawl_isic_future_import_is_valid(self) -> None:
        source = ROOT / "src" / "crawl_isic_meta.py"
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        imports = [
            node
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module == "__future__"
        ]
        self.assertEqual([alias.name for node in imports for alias in node.names], ["annotations"])

    def test_project_metadata_pins_cuda_torch_and_supported_python(self) -> None:
        metadata_path = ROOT / "pyproject.toml"
        self.assertTrue(metadata_path.is_file(), "pyproject.toml must define the environment")
        project = metadata_path.read_text(encoding="utf-8")

        self.assertIn('requires-python = ">=3.12,<3.14"', project)
        self.assertIn('"torch==2.7.0"', project)
        self.assertIn('"pytest>=8.3,<9"', project)
        self.assertIn('"ruff>=0.11,<1"', project)
        self.assertIn('torch = { index = "pytorch-cu126" }', project)
        self.assertIn('url = "https://download.pytorch.org/whl/cu126"', project)
        self.assertIn("explicit = true", project)

    def test_activation_script_enforces_home_environment_and_python_range(self) -> None:
        activation_path = ROOT / "scripts" / "server" / "activate_env.sh"
        self.assertTrue(activation_path.is_file(), "the shared activation script must exist")
        activation = activation_path.read_text(encoding="utf-8")
        self.assertIn('EXPECTED_HOME="/home/qingchan"', activation)
        self.assertIn('ENV_PREFIX="$EXPECTED_HOME/miniforge3/envs/conceptflow"', activation)
        self.assertIn('export UV_PROJECT_ENVIRONMENT="$ENV_PREFIX"', activation)
        self.assertIn('export PATH="$EXPECTED_HOME/.local/bin:$PATH"', activation)
        self.assertIn('source "$CONDA_SH" || return 1', activation)
        self.assertIn('conda activate "$ENV_PREFIX" || return 1', activation)
        self.assertIn("(3, 12) <= sys.version_info[:2] < (3, 14)", activation)

    def test_research_scripts_use_only_the_shared_activation_entrypoint(self) -> None:
        for name in RESEARCH_SCRIPTS:
            body = (ROOT / "src" / name).read_text(encoding="utf-8")
            with self.subTest(script=name):
                self.assertIn('source "$PWD/scripts/server/activate_env.sh" || exit 1', body)
                self.assertNotIn("conda activate base", body)
                self.assertNotIn("conda shell.bash hook", body)

    def test_git_attributes_define_platform_line_endings(self) -> None:
        attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
        self.assertIn("*.sh text eol=lf", attributes)
        self.assertIn("*.py text eol=lf", attributes)
        self.assertIn("*.ps1 text eol=crlf", attributes)


if __name__ == "__main__":
    unittest.main()
