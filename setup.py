from pathlib import Path

from setuptools import find_packages, setup

REPO_ROOT = Path(__file__).resolve().parent
REQS = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
INSTALL_REQUIRES = [r.strip() for r in REQS if r.strip() and not r.startswith("#")]

setup(
    name="splash",
    version="2.0.0",
    description=(
        "YOLOv12-based drowning detection for the Uninorte training pool. "
        "Fork of H20Saver, refactored for NVIDIA RTX hardware and a two-stage "
        "training workflow (public dataset -> Uninorte fine-tune)."
    ),
    long_description=(REPO_ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    python_requires=">=3.10",
    include_package_data=True,
)
