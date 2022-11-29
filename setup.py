from setuptools import setup, find_packages
import os


NAME = "jaxman"


def _get_version():
    folder = os.path.dirname(os.path.realpath(__file__))

    with open(os.path.join(folder, f"{NAME}/__init__.py"), "r") as f:
        version_line = next(line for line in f.readlines() if line.strip().startswith("__version__"))
        version = version_line.split("=")[-1].strip().replace('"', "")

    return version.strip()


with open("requirements.txt", "r") as f:
    install_requires = [p.strip() for p in f]

with open("test_requirements.txt", "r") as f:
    test_requires = [p.strip() for p in f] + install_requires


setup(
    name=NAME,
    version=_get_version(),
    author="Victor Gruselius",
    author_email="victor.gruselius@gmail.com",
    description="Vectorized Kalman filters in JAX",
    packages=find_packages(include=f"{NAME}.*"),
    install_requires=install_requires,
    tests_require=test_requires,
    python_requires=">=3.6.0",
    license_files=("LICENSE",),
    license="MIT"
)