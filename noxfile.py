"""Nox sessions."""
import nox
from nox.sessions import Session

package = "opr"


def install_cpu_torch(session: Session) -> None:
    """Install the CPU version of PyTorch."""
    session.install(
        "torch==1.12.1+cpu",
        "torchvision==0.13.1+cpu",
        "--extra-index-url",
        "https://download.pytorch.org/whl/cpu",
    )


def install_minkowskiengine(session: Session) -> None:
    """Install the MinkowskiEngine."""
    session.install("git+https://github.com/NVIDIA/MinkowskiEngine", "--no-deps")


@nox.session(python=["3.8", "3.9", "3.10", "3.11"])
def tests(session: Session) -> None:
    """Run the test suite."""
    args = session.posargs or ["--cov"]
    install_cpu_torch(session)
    install_minkowskiengine(session)
    session.install(".")
    session.install("-r", "requirements-dev.txt")
    session.run("pytest", *args)
