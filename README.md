# copydetect

## Overview
This repo is source code plagiarism detection tool.

Note that, like MOSS, plagiarism_check is designed to detect likely instances of plagiarism; it is not guaranteed to catch cheaters dedicated to evading it, and it does not provide a guarantee that plagiarism has occurred.

## Installation
plagiarism_check can be installed using `pip install plagiarism_check`. Note that Python version 3.6 or greater is required.

## Usage
If the files you want to compare to are different from the files you want to check for plagiarism (for example, if you want to also compare to submissions from previous semesters), use `-r` to provide a list of reference directories. For example, `plagiarism_check -t PA01_F20 -r PA01_F20 PA01_S20 PA01_F19`. To avoid matches with code that was provided to students, use `-b` to specify a list of directories containing boilerplate code.

>>> from plagiarism_check import CopyDetector
>>> detector = CopyDetector(test_dirs=["tests"], extensions=["py"],
...                         display_t=0.5)
>>> detector.add_file("copydetect/utils.py")
>>> detector.run()
  0.00: Generating file fingerprints
   100%|████████████████████████████████████████████████████| 8/8
  0.31: Beginning code comparison
   100%|██████████████████████████████████████████████████| 8/8
  0.31: Code comparison completed
>>> detector.generate_html_report()
Output saved to report/report.html
```
