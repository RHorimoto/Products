# ホリモトの GitHub

<div id="top"></div>


## 使用技術一覧

<!-- シールド一覧 -->
<p style="display: inline">
  <!-- 使用言語・開発環境一覧 -->
  <img src="https://img.shields.io/badge/-Python-F2C63C.svg?logo=python&style=for-the-badge">
  <img src="https://img.shields.io/badge/-Html5-E34F26.svg?logo=html5&style=social">
  <img src="https://img.shields.io/badge/-Ubuntu-E95420.svg?logo=ubuntu&style=plastic">
  <img src="https://img.shields.io/badge/-Docker-1488C6.svg?logo=docker&style=for-the-badge">
  <img src="https://img.shields.io/badge/-githubactions-FFFFFF.svg?logo=github-actions&style=for-the-badge">
</p>

## 目次

1. [プロジェクトについて](#プロジェクトについて)
2. [ディレクトリ構成](#ディレクトリ構成)


## プロジェクトについて

### 【研究】Knowledge Graph (KG)

<details>

<summary>言語モデル BERT の Masked Language Modeling を用いた KG の自動補完手法</summary>

<![KGC model1 with MLM of BERT](https://myoctocat.com/assets/images/base-octocat.svg)>

</details>

<details>

<summary>文章からの KG 自動生成およびその可視化による文章理解補助</summary>

bbbb

</details>

<details>

<summary>KG の質問システムへの応用</summary>

cccc

</details>

### 【Products】Hobby

<details>

<summary>HTML の勉強</summary>

aaaa

</details>

<p align="right">(<a href="#top">トップへ</a>)</p>


## ディレクトリ構成

<pre>
.
├── Products
│   ├── assets
│   │   ├── ironman.jpg
│   │   └── spiderman.jpg
│   ├── marvel.html
│   └── pages
│       └── Ironman.html
├── README.md
└── 研究
    ├── JSAI2024
    │   ├── 2024_JSAI.pptx
    │   └── JSAI_2024.pdf
    └── src_MLM_tail_predict
        ├── data
        │   ├── FB13
        │   │   ├── dev.tsv
        │   │   ├── entities.txt
        │   │   ├── entity2text.txt
        │   │   ├── entity2text_capital.txt
        │   │   ├── entity2textshort.txt
        │   │   ├── relation2text.txt
        │   │   ├── relations.txt
        │   │   ├── test.tsv
        │   │   └── train.tsv
        │   ├── FB15K
        │   │   ├── FB15k_mid2description.txt
        │   │   ├── FB15k_mid2name.txt
        │   │   ├── dev.tsv
        │   │   ├── entities.txt
        │   │   ├── entity2id.txt
        │   │   ├── entity2text.txt
        │   │   ├── entity2textlong.txt
        │   │   ├── relation2id.txt
        │   │   ├── relation2text.txt
        │   │   ├── relations.txt
        │   │   ├── test.tsv
        │   │   ├── test2id.txt
        │   │   ├── train.tsv
        │   │   ├── train2id.txt
        │   │   └── valid2id.txt
        │   ├── FB15k-237
        │   │   ├── FB15k_mid2description.txt
        │   │   ├── FB15k_mid2name.txt
        │   │   ├── dev.tsv
        │   │   ├── entities.txt
        │   │   ├── entity2text.txt
        │   │   ├── entity2textlong.txt
        │   │   ├── relation2text.txt
        │   │   ├── relations.txt
        │   │   ├── test.tsv
        │   │   └── train.tsv
        │   ├── WN11
        │   │   ├── dev.tsv
        │   │   ├── entities.txt
        │   │   ├── entity2text.txt
        │   │   ├── relation2text.txt
        │   │   ├── relations.txt
        │   │   ├── test.tsv
        │   │   ├── train.tsv
        │   │   ├── train_1.tsv
        │   │   ├── train_10.tsv
        │   │   ├── train_15.tsv
        │   │   ├── train_20.tsv
        │   │   └── train_5.tsv
        │   ├── WN18
        │   │   ├── dev.tsv
        │   │   ├── entities.txt
        │   │   ├── entity2id.txt
        │   │   ├── entity2text.txt
        │   │   ├── relation2id.txt
        │   │   ├── relation2text.txt
        │   │   ├── relations.txt
        │   │   ├── test.tsv
        │   │   ├── test2id.txt
        │   │   ├── train.tsv
        │   │   ├── train2id.txt
        │   │   ├── valid2id.txt
        │   │   └── wordnet-mlj12-definitions.txt
        │   ├── WN18RR
        │   │   ├── all_triples.txt
        │   │   ├── dev.tsv
        │   │   ├── dev_triples.txt
        │   │   ├── entities.txt
        │   │   ├── entity2text.txt
        │   │   ├── relation2text.txt
        │   │   ├── relations.txt
        │   │   ├── test.tsv
        │   │   ├── test_triples.txt
        │   │   ├── train.tsv
        │   │   ├── train_dev_triples.txt
        │   │   ├── train_triples.txt
        │   │   └── wordnet-mlj12-definitions.txt
        │   ├── YAGO3-10
        │   │   ├── dev.tsv
        │   │   ├── entities.txt
        │   │   ├── entity2text.txt
        │   │   ├── relation2text.txt
        │   │   ├── relations.txt
        │   │   ├── test.tsv
        │   │   └── train.tsv
        │   └── umls
        │       ├── dev.tsv
        │       ├── entities.txt
        │       ├── entity2text.txt
        │       ├── entity2textlong.txt
        │       ├── relation2text.txt
        │       ├── relations.txt
        │       ├── test.tsv
        │       └── train.tsv
        ├── e.sh
        ├── mlm_tail.py
        └── requirements.txt
</pre>

<!-- README 作成方法の参考ページのリンク -->
<br />
<div align="right">
    <a href="https://qiita.com/shun198/items/c983c713452c041ef787"><strong>README 作成方法の参考ページ »</strong></a>
</div>
<br />

<p align="right">(<a href="#top">トップへ</a>)</p>

