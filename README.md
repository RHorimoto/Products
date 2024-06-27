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

### 【studies】Knowledge Graph (KG)

<details>

<summary>言語モデル BERT の Masked Language Modeling を用いた KG の自動補完手法</summary>

[2024 年度人工知能学会全国大会 (第 38 回)](https://www.ai-gakkai.or.jp/jsai2024/) で発表.

<!-- 2024 年度人工知能学会全国大会 (第 38 回) のリンク -->
<br />
<div align="left">
    <a href="https://github.com/RHorimoto/Products/tree/main/studies/KGC_MLM/JSAI2024"><strong>発表資料 »</strong></a>
</div>
<br />

* MLM を利用した BERT の Fine-tuning 手法

![Fine-tuning of MLM](https://github.com/RHorimoto/Products/blob/main/studies/KGC_MLM/assets/MLM_fine-tuning.png)

* KG の自動補完手法 model1

![KGC model1 with MLM of BERT](https://github.com/RHorimoto/Products/blob/main/studies/KGC_MLM/assets/model1.png)

* KG の自動補完手法 model2

![KGC model2 with MLM of BERT](https://github.com/RHorimoto/Products/blob/main/studies/KGC_MLM/assets/model2.png)

モデルの改善を目標に研究を続行. 

</details>

<details>

<summary>文章からの KG 自動生成およびその可視化による文章理解補助</summary>

現在研究中. 

[ナレッジグラフ推論チャレンジ 2024](https://challenge.knowledge-graph.jp/2024/) に参加予定.

</details>

<details>

<summary>KG の質問システムへの応用</summary>

Graph Retrieval Augmented Generation (Graph RAG) として KG の応用を検討中. 

</details>

### 【Products】Hobby

<details>

<summary>HTML の勉強</summary>

趣味の映画鑑賞 (MARVEL movies) について web ページを作成中. 

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
    └── KGC_MLM
        ├── JSAI2024
        │   ├── 2024_JSAI.pptx
        │   └── JSAI_2024.pdf
        ├── assets
        │   ├── MLM_fine-tuning.png
        │   ├── model1.png
        │   └── model2.png
        └── src
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

