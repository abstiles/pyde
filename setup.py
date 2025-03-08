from setuptools import setup

setup(
    name='pyde',
    packages=['pyde'],
    python_requires=">=3.9",
    install_requires=['pyyaml>=6.0','markdown-it-py[plugins]>=3.0','Jinja2>=3.1'],
    extras_require={},
    zip_safe=False,
    entry_points={"console_scripts": [
        "pyde=pyde.cli:main",
    ]},
)
