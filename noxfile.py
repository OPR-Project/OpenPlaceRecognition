"""Nox sessions."""
import nox
from nox.sessions import Session

nox.options.sessions = "tests", "lint"

PYTHON_VERSIONS = ("3.8", "3.9", "3.10")
PYTORCH_VERSIONS = ("1.12.1", "1.13.1", "2.0.1")
TORCHVISION_VERSIONS_DICT = {
    "1.12.1": "0.13.1",
    "1.13.1": "0.14.1",
    "2.0.1": "0.15.2",
}

locations = "src", "tests", "noxfile.py"
package = "opr"


def install_cpu_torch(session: Session, pytorch: str = "1.12.1") -> None:
    """Install the CPU version of PyTorch."""
    session.install(
        f"torch=={pytorch}+cpu",
        f"torchvision=={TORCHVISION_VERSIONS_DICT[pytorch]}+cpu",
        "--extra-index-url",
        "https://download.pytorch.org/whl/cpu",
    )


def install_minkowskiengine(session: Session) -> None:
    """Install the MinkowskiEngine."""
    session.install("setuptools==68.0.0")
    session.run("pip", "install", "git+https://github.com/NVIDIA/MinkowskiEngine", "--no-deps")


@nox.session
@nox.parametrize(
    "python,pytorch",
    [
        (python, pytorch)
        for python in PYTHON_VERSIONS
        for pytorch in PYTORCH_VERSIONS
        if (python, pytorch) not in (("3.11", "1.12.1"), ("3.11", "1.13.1"))
    ],
)
def tests(session: Session, pytorch: str) -> None:
    """Run the test suite."""
    args = session.posargs or ["--cov"]
    install_cpu_torch(session, pytorch)
    install_minkowskiengine(session)
    session.install("-e", ".")
    session.install("-r", "requirements-dev.txt")
    session.run("pytest", *args)


@nox.session
@nox.parametrize("python", PYTHON_VERSIONS)
def lint(session: Session) -> None:
    """Lint using flake8."""
    args = session.posargs or locations
    install_cpu_torch(session)
    install_minkowskiengine(session)
    session.install("-e", ".")
    session.install("-r", "requirements-dev.txt")
    session.run("flake8", *args)
