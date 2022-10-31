# Modifying, DO NOT release now!!!

CofeNet
==============================================
This is the source code of COLING 2022 paper "CofeNet: Context and Former-Label Enhanced Net for Complicated Quotation Extraction". See our [paper](https://aclanthology.org/2022.coling-1.215/) for more details.

**Abstract**: Quotation extraction aims to extract quotations from written text. There are three components in a quotation: _source_ refers to the holder of the quotation, _cue_ is the trigger word(s), and _content_ is the main body. Existing solutions for quotation extraction mainly utilize rulebased approaches and sequence labeling models. While rule-based approaches often lead to low recalls, sequence labeling models cannot well handle quotations with complicated structures. In this paper, we propose the **Co**ntext and **F**ormer-Label **E**nhanced **Net** (CofeNet) for quotation extraction. CofeNet is able to extract complicated quotations with components of variable lengths and complicated structures. On two public datasets (_i.e._, PolNeAR and Riqua) and one proprietary dataset (_i.e._, PoliticsZH), we show that our CofeNet achieves state-of-the-art performance on complicated quotation extraction.
<p align="center"><img width="100%" src="docs/architecture.png" /></p>

# Setup
* Environment settings
```
# Python version==3.7
git clone https://github.com/cofe-ai/CofeNet.git
cd CofeNet
pip install -r requirements.txt
```

* Data resource settings

TODO: 数据集下载和配置

# Run

TODO: 快速执行代码：训练 & 评估

# Experiment
CofeNet Detail Experiment [here](./docs/cofenet-detail-exp.pdf)

# Cite

If the code or data help you, please cite the following paper.

```
@inproceedings{wang-etal-2022-cofenet,
    title = "{C}ofe{N}et: Context and Former-Label Enhanced Net for Complicated Quotation Extraction",
    author = "Wang, Yequan  and
      Li, Xiang  and
      Sun, Aixin  and
      Meng, Xuying  and
      Liao, Huaming  and
      Guo, Jiafeng",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.215",
    pages = "2438--2449",
    abstract = "Quotation extraction aims to extract quotations from written text. There are three components in a quotation: \textit{source} refers to the holder of the quotation, \textit{cue} is the trigger word(s), and \textit{content} is the main body. Existing solutions for quotation extraction mainly utilize rule-based approaches and sequence labeling models. While rule-based approaches often lead to low recalls, sequence labeling models cannot well handle quotations with complicated structures. In this paper, we propose the \textbf{Co}ntext and \textbf{F}ormer-Label \textbf{E}nhanced \textbf{Net} () for quotation extraction. is able to extract complicated quotations with components of variable lengths and complicated structures. On two public datasets (and ) and one proprietary dataset (), we show that our achieves state-of-the-art performance on complicated quotation extraction.",
}
```
