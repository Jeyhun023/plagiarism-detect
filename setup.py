from setuptools import setup, Extension

with open("README.md", "r") as readme_fp:
    readme = readme_fp.read()

# NOTE
# the C extension is not currently built for versions uploaded to PyPI.
# Speedup is not meaningful and it makes cross-platform support quite
# a bit more painful

setup(name="abyss-plagiarism-detect",
      author="Jeyhun Rashidov",
      author_email="creshidov23@gmail.com",
      version="0.1.1",
      description="Code plagiarism detection tool",
      long_description=readme,
      long_description_content_type="text/markdown",
      url="https://github.com/Jeyhun023/plagiarism-detect",
      packages=["plagiarismdetect"],
      ext_modules=[Extension("plagiarismdetect.winnow",
                             sources=["plagiarismdetect/winnow/winnow.c"],
                             optional=True)],
      install_requires=["numpy", "matplotlib", "jinja2", "pygments", "tqdm"],
      package_data={"plagiarismdetect" : ["data/*"]},
      python_requires=">=3.6",
      entry_points={"console_scripts" : [
          "plagiarismdetect = plagiarismdetect.__main__:main"]},
      classifiers=[
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Python :: 3.9",
          "Programming Language :: Python :: 3.10",
          "Programming Language :: Python :: 3 :: Only",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
          "Development Status :: 4 - Beta",
          "Intended Audience :: Education"
      ])
