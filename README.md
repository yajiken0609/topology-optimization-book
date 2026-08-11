# トポロジー最適化入門 — 公式サンプルコード

本リポジトリは、矢地謙太郎 著『トポロジー最適化入門』の公式サンプルコードです。

本書では、連続最適化、逐次凸計画法、有限要素解析、感度解析、密度法によるトポロジー最適化、弾性問題、3次元問題、積層造形を考慮した設計までを、Pythonによる実装とともに解説します。

- 書籍公式サイト: https://www.yajiken.jp/book
- 著者サイト: https://www.yajiken.jp

> **Note**  本リポジトリのコードは教育・学習を目的としたサンプルです。実務設計における性能、安全性、妥当性を保証するものではありません。

## Contents

| 章 | 題目 | Notebook |
|---|---|---|
| 第2章 | 連続最適化 | [`ch02.ipynb`](ch02.ipynb) |
| 第3章 | 逐次凸計画法 | [`ch03.ipynb`](ch03.ipynb) |
| 第4章 | 有限要素解析 | [`ch04.ipynb`](ch04.ipynb) |
| 第5章 | トポロジー最適化問題の定式化 | [`ch05.ipynb`](ch05.ipynb) |
| 第6章 | 感度解析 | [`ch06.ipynb`](ch06.ipynb) |
| 第7章 | 実践的なテクニック | [`ch07.ipynb`](ch07.ipynb) |
| 第8章 | 弾性問題への展開 | [`ch08.ipynb`](ch08.ipynb) |
| 第9章 | 3次元問題への展開 | [`ch09.ipynb`](ch09.ipynb) |

第1章「トポロジー最適化とは」には対応するNotebookはありません。

## Supporting modules

Notebookから利用する補助モジュールです。原則としてNotebookと同じディレクトリに置いたまま実行してください。

- `optimizer.py` — 最適化アルゴリズム
- `filtering.py` — 2次元問題のフィルタリング
- `filtering3d.py` — 3次元問題のフィルタリング
- `sens_heat.py` — 2次元熱伝導問題
- `sens_heat3d.py` — 3次元熱伝導問題
- `sens_elastic.py` — 2次元弾性問題
- `sens_elastic_compmech.py` — コンプライアントメカニズム

## Installation

Python環境を用意した後、本リポジトリを取得してください。

```bash
git clone https://github.com/yajiken/topology-optimization-book.git
cd topology-optimization-book
```

必要なライブラリをインストールします。

```bash
python -m pip install -r requirements.txt
```

JupyterLabを起動します。

```bash
jupyter lab
```

対応する章のNotebookを開き、上から順にセルを実行してください。

## Dependencies

主に以下のPythonパッケージを使用します。

- NumPy
- SciPy
- Matplotlib
- PyAMG
- PyVista
- JupyterLab

第9章の3次元問題では `pyamg` および `pyvista` を使用します。また、3次元例題は計算規模が大きいため、計算機環境によっては長い計算時間や多くのメモリを必要とします。まず動作を確認したい場合は、要素数を小さくして実行してください。

## Relationship to the book

本リポジトリは「完成したコードをコピーして使う」ことだけを目的としていません。書籍本文を読みながら、数式とコードの対応を確認し、自分で実装していくことを推奨します。

書籍の正誤表、補足資料、更新情報は公式サイトをご覧ください。

https://www.yajiken.jp/book

## Issues

コードの不具合や、書籍との対応に関する技術的な問題は [GitHub Issues](../../issues) から報告してください。

書籍本文の正誤情報については、公式サイトの正誤表を最新版とします。

## Citation

研究・教育等で本書または本リポジトリを参照する場合は、書籍を引用してください。引用情報は [`CITATION.cff`](CITATION.cff) にも記載しています。

## License

ソースコードは [BSD 3-Clause License](LICENSE) のもとで公開します。

Copyright (c) 2026 Kentaro Yaji
