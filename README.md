# Semantic Segmentation Toolbox and Benchmark for Sonar Image


[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mmsegmentation)](https://pypi.org/project/mmsegmentation/)
[![PyPI](https://img.shields.io/pypi/v/mmsegmentation)](https://pypi.org/project/mmsegmentation)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmsegmentation.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmsegmentation/workflows/build/badge.svg)](https://github.com/open-mmlab/mmsegmentation/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmsegmentation/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmsegmentation)
[![license](https://img.shields.io/github/license/open-mmlab/mmsegmentation.svg)](https://github.com/open-mmlab/mmsegmentation/blob/main/LICENSE)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmsegmentation.svg)](https://github.com/open-mmlab/mmsegmentation/issues)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmsegmentation.svg)](https://github.com/open-mmlab/mmsegmentation/issues)



<!-- English | [简体中文](README_zh-CN.md) -->

## Introduction

This project is an open source toolbox for semantic segmentation, which is maintained by [OpenMMLab](https://openmmlab.com).

## Installation

Please refer to [get_started](https://mmsegmentation.readthedocs.io/zh-cn/latest/get_started.html) for installation mmcv and the mmengine. Specifically, you can install MMSegmentation with the following command:
```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```
Then install this project:
```shell
pip install -v -e .
```

## Get Started
For training, the command is as follows:
```shell
python tools/train.py configs/your/config/file
```

## License

This project is released under the [Apache 2.0 license](LICENSE).
