# -*- coding: utf-8 -*-

"""
setup.py implementation, interesting because it parsed the first __init__.py and
    extracts the `__author__` and `__version__`
"""

import sys
from ast import Assign, Name, parse
from operator import attrgetter
from os import path
from os.path import extsep

from setuptools import find_packages, setup

if sys.version_info[:2] >= (3, 12):
    from ast import Del as Str
else:
    from ast import Str

    if sys.version_info[0] == 2:
        from itertools import ifilter as filter
        from itertools import imap as map

if sys.version_info[:2] > (3, 7):
    from ast import Constant
else:
    from ast import expr

    # Constant. Will never be used in Python =< 3.8
    Constant = type("Constant", (expr,), {})


package_name = "e4s2024"

with open(
    path.join(path.dirname(__file__), "README{extsep}md".format(extsep=extsep)),
    "rt",
    encoding="utf-8",
) as fh:
    long_description = fh.read()


def main():
    """Main function for setup.py; this actually does the installation"""
    with open(
        path.join(
            path.abspath(path.dirname(__file__)),
            package_name,
            "__init__{extsep}py".format(extsep=extsep),
        )
    ) as f:
        parsed_init = parse(f.read())

    __author__, __version__, __description__ = map(
        lambda node: node.value if isinstance(node, Constant) else node.s,
        filter(
            lambda node: isinstance(node, (Constant, Str)),
            map(
                attrgetter("value"),
                filter(
                    lambda node: isinstance(node, Assign)
                    and any(
                        filter(
                            lambda name: isinstance(name, Name)
                            and name.id
                            in frozenset(
                                ("__author__", "__version__", "__description__")
                            ),
                            node.targets,
                        )
                    ),
                    parsed_init.body,
                ),
            ),
        ),
    )

    setup(
        name=package_name,
        author=__author__,
        author_email="1398159954@qq.com",
        version=__version__,
        url="https://github.com/e4s2024/E4S2024",
        description=__description__,
        long_description=long_description,
        long_description_content_type="text/markdown",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Environment :: Console",
            "Intended Audience :: Developers",
            "License :: OSI Approved",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Programming Language :: Python :: 3.13",
            "Programming Language :: Python :: Implementation :: CPython"
        ],
        license="MIT",
        license_files=["LICENSE"],
        packages=find_packages(),
        python_requires=">=3.9",
    )


def setup_py_main():
    """Calls main if `__name__ == '__main__'`"""
    if __name__ == "__main__":
        main()


setup_py_main()
