from setuptools import Extension, setup
from Cython.Build import cythonize

# Define the Extension module configuration
extensions = [
    Extension(
        name="user_agent_parser",
        sources=["user_agent_parser.pyx"],
    )
]

setup(
    name="user_agent_parser",
    version="0.1.0",
    description="Cythonized User Agent Parser with Quantum & Qubic support",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "embedsignature": True,
        },
    ),
)
