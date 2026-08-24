"""Install a stub mamba_ssm package into the current environment.

The Mage-VL checkpoint's remote code references streammind_gate.py, whose
top-level `import mamba_ssm` fails transformers' static import check on
macOS (mamba-ssm has no macOS build). At runtime the streaming gate is
lazily imported and unused for the image path, so an empty stub suffices.
"""

import site
from pathlib import Path

root = Path(site.getsitepackages()[0]) / "mamba_ssm"
(root / "models").mkdir(parents=True, exist_ok=True)
(root / "__init__.py").write_text(
    "# stub for transformers check_imports; streaming gate unused for images\n"
)
(root / "models" / "__init__.py").write_text("")
(root / "models" / "mixer_seq_simple.py").write_text(
    'def create_block(*args, **kwargs):\n'
    '    raise ImportError("mamba_ssm stub: streaming gate unavailable")\n'
)
print(f"stub installed at {root}")
