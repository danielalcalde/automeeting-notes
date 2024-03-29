
from setuptools import setup, find_packages

setup(
    name="automeeting-notes",
    packages=find_packages(),
    version="1.0",
    description="An Python webui for generating speaker-diarized meeting transcriptions from audio recordings",
    readme="README.md",
    python_requires=">=3.7",
    author="danielalcalde",
    url="https://github.com/danielalcalde/automeeting-notes",
    license="MIT",
    scripts=["automeeting", "automeeting-webui"],
    install_requires=["numpy", "torch>=1.8.0", "torchvision", "openai-whisper", "pyannote.audio", "gradio<4.0"]
)
