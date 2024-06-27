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

<!-- README 作成方法の参考ページのリンク -->
<br />
<div align="right">
    <a href="https://qiita.com/shun198/items/c983c713452c041ef787"><strong>README 作成方法の参考ページ »</strong></a>
</div>
<br />


## プロジェクトについて

###【研究】Knowledge Graph (KG)

<details>

<summary>言語モデル BERT の Masked Language Modeling を用いた KG の自動補完手法</summary>

aaaa

<summary>文章からの KG 自動生成およびその可視化による文章理解補助</summary>

bbbb

<summary>KG の質問システムへの応用</summary>

cccc

</details>

###【Products】Hobby

<details>

<summary>HTML の勉強</summary>

aaaa

</details>

<p align="right">(<a href="#top">トップへ</a>)</p>


## ディレクトリ構成

<!-- Treeコマンドを使ってディレクトリ構成を記載 -->

❯ tree -a -I "node_modules|.next|.git|.pytest_cache|static" -L 2
.
├── .devcontainer
│   └── devcontainer.json
├── .env
├── .github
│   ├── action
│   ├── release-drafter.yml
│   └── workflows
├── .gitignore
├── Makefile
├── README.md
├── backend
│   ├── .vscode
│   ├── application
│   ├── docs
│   ├── manage.py
│   ├── output
│   ├── poetry.lock
│   ├── project
│   └── pyproject.toml
├── containers
│   ├── django
│   ├── front
│   ├── mysql
│   └── nginx
├── docker-compose.yml
├── frontend
│   ├── .gitignore
│   ├── README.md
│   ├── __test__
│   ├── components
│   ├── features
│   ├── next-env.d.ts
│   ├── package-lock.json
│   ├── package.json
│   ├── pages
│   ├── postcss.config.js
│   ├── public
│   ├── styles
│   ├── tailwind.config.js
│   └── tsconfig.json
└── infra
    ├── .gitignore
    ├── docker-compose.yml
    ├── main.tf
    ├── network.tf
    └── variables.tf

<p align="right">(<a href="#top">トップへ</a>)</p>

