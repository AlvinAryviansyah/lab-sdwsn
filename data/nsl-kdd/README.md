# NSL-KDD dataset

File mentah (`KDDTrain+.txt`, `KDDTest+.txt`) tidak di-commit ke git karena
ukurannya ~22 MB dan tersedia bebas. Download ulang dengan:

```bash
cd data/nsl-kdd
for f in KDDTrain+.txt KDDTest+.txt; do
  curl -sL "https://raw.githubusercontent.com/HoaNP/NSL-KDD-DataSet/master/$f" -o "$f"
done
```

Dipakai oleh `ml/bigan_ids.py` (intrusion detection) dan diproses oleh
`ml/nslkdd_data.py`. NSL-KDD disebut di future work paper (Sec. VI).
