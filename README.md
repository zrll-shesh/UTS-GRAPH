# UTS Graf Sains Data: Neo4j, Vector Space Model, dan Rekomendasi KNN

**Nazril Ravi Pratama** | NIM 24031554129 | Kelas International 24 | S1 Sains Data, Universitas Negeri Surabaya

Dataset: [RetailRocket e-commerce](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) (`events.csv`, Roman Zykov).

## Isi repositori

```
UTS_Graf_Sains_Data.ipynb   notebook utama: jawaban, penjelasan, hasil, visualisasi
src/
  config.py                 semua pengaturan (periode P1/P2, bobot, seed = NIM, koneksi)
  data_prep.py              baca events.csv, filter periode, simpan .csv
  queries.py                seluruh query Cypher (dicetak dan dijelaskan di notebook)
  graph_io.py               koneksi Neo4j, impor graf, baca/tulis hasil (juga bisa dijalankan sebagai CLI)
  vsm.py                    matriks interaksi, Vector Space Model (TF-IDF), cosine top-k
  recommender.py            KNN berbasis user/item dan baseline popularity
  metrics.py                ground truth, Hit@N, NDCG@N, bootstrap CI
  viz.py                    seluruh grafik (disimpan ke figures/)
tests/test_core.py          tes kecil untuk perhitungan inti (pytest)
data/processed/             events_p1_train.csv dan events_p2_test.csv (dibuat oleh notebook)
figures/                    PNG hasil visualisasi (dibuat oleh notebook)
```
