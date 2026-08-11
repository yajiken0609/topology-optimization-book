# 『トポロジー最適化入門』サンプルコード

本リポジトリは、オーム社刊  
**『トポロジー最適化入門』**  
の公式Pythonサンプルコードです。

本書では、連続最適化、逐次凸計画法、有限要素法、感度解析から、
トポロジー最適化の実装、さらに弾性問題・3次元問題への展開までを、
Pythonコードとともに解説しています。

📘 **著者による書籍公式サイト**  
https://www.yajiken.jp/book

---

## サンプルコード

各Jupyter Notebookは、書籍の各章に対応しています。

| 章 | 内容 | Notebook |
|---|---|---|
| 第2章 | 連続最適化 | [ch02.ipynb](ch02.ipynb) |
| 第3章 | 逐次凸計画法 | [ch03.ipynb](ch03.ipynb) |
| 第4章 | 有限要素解析 | [ch04.ipynb](ch04.ipynb) |
| 第5章 | トポロジー最適化問題の定式化 | [ch05.ipynb](ch05.ipynb) |
| 第6章 | 感度解析 | [ch06.ipynb](ch06.ipynb) |
| 第7章 | 実践的なテクニック | [ch07.ipynb](ch07.ipynb) |
| 第8章 | 弾性問題への展開 | [ch08.ipynb](ch08.ipynb) |
| 第9章 | 3次元問題への展開 | [ch09.ipynb](ch09.ipynb) |

第1章「トポロジー最適化とは」には対応するサンプルコードはありません。

---

## 動作確認環境

本書および本リポジトリのコードは、以下の環境で動作を確認しています。

| Software / Library | Version |
|---|---:|
| Python | 3.13.5 |
| NumPy | 2.1.3 |
| SciPy | 1.15.3 |
| Matplotlib | 3.10.0 |
| pyAMG* | 5.3.0 |
| PyVista* | 0.46.4 |

\* pyAMGおよびPyVistaは、第9章「3次元問題への展開」で使用します。

Pythonについては、3.9系以降でも動作を確認しています。
詳細については、書籍付録「Python環境の設定」を参照してください。

---

## インストール

Pythonがインストールされた環境で、必要なライブラリをインストールします。

```bash
pip install -r requirements.txt
```

`requirements.txt` には以下のライブラリが含まれています。

- NumPy
- SciPy
- Matplotlib
- pyAMG
- PyVista
- JupyterLab

第2章から第8章までは、主にNumPy、SciPy、Matplotlibを使用します。
第9章の3次元問題では、追加でpyAMGおよびPyVistaを使用します。

---

## 実行方法

リポジトリを取得します。

```bash
git clone https://github.com/yajiken/topology-optimization-book.git
cd topology-optimization-book
```

JupyterLabを起動します。

```bash
jupyter lab
```

ブラウザ上で、読みたい章に対応するNotebookを開いて実行してください。

例えば `ch07.ipynb` は、第7章「実践的なテクニック」に対応しています。

本書ではJupyterLabの利用を想定していますが、
VS CodeやGoogle Colabなど、Jupyter Notebook形式（`.ipynb`）を
実行できる他の環境を利用しても問題ありません。

---

## Pythonスクリプト

Notebookに加えて、有限要素解析、感度解析、フィルタリング、
最適化アルゴリズムなどに使用するPythonスクリプトを収録しています。

```text
optimizer.py
filtering.py
filtering3d.py
sens_heat.py
sens_heat3d.py
sens_elastic.py
sens_elastic_compmech.py
```

これらのスクリプトは各Notebookから読み込んで使用します。
NotebookとPythonスクリプトは同じディレクトリに配置した状態で実行してください。

---

## 第9章の実行について

第9章では3次元問題を扱うため、第2章から第8章までの例題と比較して
計算時間および必要なメモリが大幅に増加します。

使用する計算機環境によっては、計算に時間がかかったり、
メモリが不足したりする場合があります。

動作確認を目的とする場合は、必要に応じて解析モデルの要素数を
小さくして実行してください。

---

## 書籍について

**『トポロジー最適化入門』**  
矢地謙太郎 著  
オーム社

書籍情報、正誤表、補足資料などは、以下の公式サイトで公開します。

**https://www.yajiken.jp/book**

---

## 引用について

本リポジトリのコードを研究・教育・出版物等で利用し、引用が必要な場合は、
以下の書籍を引用してください。

> 矢地謙太郎，『トポロジー最適化入門』，オーム社，2026．

---

## Issues

コードの不具合を発見された場合は、
[Issues](../../issues) からお知らせください。

書籍本文の誤植・訂正については、
書籍公式サイトの正誤表をご確認ください。

---

## 利用上の注意

本リポジトリで公開しているコードは、
『トポロジー最適化入門』の内容を理解するための
教育・学習目的のサンプルコードです。

コードの利用によって得られる解析結果や最適化結果について、
実務設計における性能、安全性、妥当性を保証するものではありません。

---

## ライセンス

本リポジトリのソースコードは
[BSD 3-Clause License](LICENSE)
のもとで公開しています。

---

## 著者

**矢地謙太郎 / Kentaro Yaji**

🌐 https://www.yajiken.jp
