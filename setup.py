from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    readme = fh.read()

packages=find_packages()
print(packages)

setup(
    name="torchquant",
    description="TorchQuant: A Hackable Quantization Toolkit",
    long_description=readme,
    version="0.1",
    packages=find_packages(include=["torchquant"]),
    author="Milad Alizadeh & Shyam Tailor",
    url="https://github.com/camlsys/torchquant",
    keywords=["pytorch", "torch", "quantization", "quantisation"],
    author_email="hello@mil.ad",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License"
    ],
    # do not specify torch / torchvision since this can break things using conda
    install_requires=["efficientnet_pytorch"],
    python_requires=">=3.6",
)
