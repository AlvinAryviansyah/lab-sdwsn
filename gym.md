# Panduan sambungan Gym: TLDQN online di ns-3

Dokumen ini memasang modul OpenGym pada ns-3.48, lalu menjalankan
`ml/tldqn_routing_gym.py` supaya Python memilih next hop dan ns-3
mengembalikan reward.

Urutan di bawah sudah memasukkan penyesuaian yang wajib pada ns-3.48
(nama target, berkas Protobuf, bentrok nama contoh, dan alias NumPy).
Ikuti langkahnya dari atas. Jangan mengaktifkan contoh ns-3
(`--enable-examples`): contoh bawaan ns3-gym juga bernama `opengym`
dan configure akan gagal.

Satu episode sama dengan satu proses ns-3 dan satu paket. Proses itu
berhenti sendiri setelah paket selesai. Tidak ada CSV yang ditulis.

## Yang perlu sudah ada

- Ubuntu, dengan Python 3.9 atau lebih baru.
- ns-3.48 yang sudah terpasang. `./ns3` ada di folder sumbernya.
- Repo ini (`lab-sdwsn`) sudah di-clone. Skrip
  `ml/tldqn_routing_gym.py` di repo ini sudah memotong state 44 angka
  menjadi 42 fitur slot. Pakai salinan itu.

Di seluruh perintah, dua variabel ini yang diganti jika path-nya berbeda:

```bash
export NS3=$HOME/ns-allinone-3.48/ns-3.48
export LAB=$NS3/lab-sdwsn
```

`LAB` adalah folder yang berisi `ml/` dan `ns3-sim/`. `NS3` adalah
folder yang berisi skrip `./ns3`.

## 1. Pustaka sistem

```bash
sudo apt update
sudo apt install -y libzmq3-dev libprotobuf-dev protobuf-compiler pkg-config
```

## 2. Modul ns3-gym

Clone langsung sebagai `contrib/ns3-gym`, lalu pindah ke cabang CMake.
Folder modulnya memang bernama `ns3-gym`. Tidak ada subfolder `opengym`
yang perlu dipindahkan.

```bash
cd "$NS3/contrib"
rm -rf ns3-gym
git clone https://github.com/tkn-tub/ns3-gym.git
cd ns3-gym
git checkout 'app-ns-3.36+'
```

`ls CMakeLists.txt model/messages.proto` harus menampilkan kedua nama
itu. Kalau tidak, clone-nya belum di cabang yang benar. Jangan lanjut.

## 3. Siapkan Protobuf dan CMakeLists

Perintah ini membuat `messages.pb.cc` serta `messages.pb.h`, lalu
mengubah `CMakeLists.txt` supaya ns-3.48 memakai kedua berkas itu.
Aman dijalankan ulang: bagian yang sudah diubah tidak digandakan.

```bash
cd "$NS3/contrib/ns3-gym"
protoc -I model --cpp_out=model model/messages.proto
mkdir -p model/ns3gym/ns3gym
protoc -I model --python_out=model/ns3gym/ns3gym model/messages.proto
python3 - << 'PY'
from pathlib import Path
p = Path("CMakeLists.txt")
t = p.read_text()
t = t.replace("model/messages.proto", "model/messages.pb.cc")
t = t.replace("${libopengym-obj}", "opengym")
if "model/messages.pb.h" not in t:
    t = t.replace("model/spaces.h", "model/spaces.h\n model/messages.pb.h", 1)
cut = t.find("# need protobuf_generate")
if cut != -1:
    t = t[:cut]
p.write_text(t)
PY
ls model/messages.pb.cc model/messages.pb.h
grep -n "messages.pb" CMakeLists.txt
```

`grep` harus memuat `messages.pb.cc` dan `messages.pb.h`. Tidak boleh
masih ada `messages.proto` atau `protobuf_generate`.

## 4. Build skrip Gym

```bash
cp "$LAB/ns3-sim/honeycomb-sdwsn-gym.cc" "$NS3/scratch/"
cd "$NS3"
./ns3 configure --disable-examples
./ns3 build honeycomb-sdwsn-gym
```

Build berhasil jika baris terakhir memuat
`Linking CXX executable` dan
`ns3.48-honeycomb-sdwsn-gym-default`, lalu prompt kembali tanpa
`CMake Error` atau `error:`.

## 5. Python

Lingkungan virtual diletakkan di dalam repo. Kalau `.venv` sudah ada,
lewati perintah `python3 -m venv`.

```bash
cd "$LAB"
python3 -m venv .venv
.venv/bin/pip install -r ml/requirements.txt
.venv/bin/pip install -e "$NS3/contrib/ns3-gym/model/ns3gym"
```

NumPy yang sekarang tidak lagi menyediakan `np.float`. Ganti alias itu
di paket yang baru dipasang:

```bash
sed -i \
  -e 's/np\.float\b/np.float64/g' \
  -e 's/np\.int\b/np.int64/g' \
  -e 's/np\.uint\b/np.uint64/g' \
  -e 's/np\.bool\b/bool/g' \
  "$NS3/contrib/ns3-gym/model/ns3gym/ns3gym/ns3env.py"
.venv/bin/python -c "from ns3gym import ns3env; print('ns3gym siap')"
```

`ns3gym siap` harus tercetak. Peringatan bahwa Gym tidak lagi dirawat
boleh muncul. Itu bukan kegagalan.

## 6. Satu episode

Dari folder ns-3, dengan `.venv` sudah aktif, perintah yang dipakai adalah:

```bash
cd lab-sdwsn/
python ml/tldqn_routing_gym.py --ns3-dir /home/vboxuser/ns-allinone-3.48/ns-3.48 --episodes 1 --num-nodes 20 --seed 1
```

Ganti `/home/vboxuser/ns-allinone-3.48/ns-3.48` jika folder ns-3 berada di tempat lain. Bentuk yang sama memakai variabel dari awal dokumen, dan tetap jalan walaupun `.venv` tidak diaktifkan:

```bash
cd "$LAB"
.venv/bin/python ml/tldqn_routing_gym.py \
  --ns3-dir "$NS3" --episodes 1 --num-nodes 20 --seed 1
```

Berhasil jika ada baris seperti:

```text
target 1: return=1.000 steps=1 delivered
Return rata-rata: 1.000
```

`delivered` berarti paket sampai ke sink. Hasil lain yang juga sah,
kalau langkah atau seed-nya berbeda: `link_loss`, `malicious_drop`,
`relay_drop`, atau `ttl`.

Lima episode, masing-masing membuka ns-3 sekali lalu menutupnya:

```bash
cd "$LAB"
.venv/bin/python ml/tldqn_routing_gym.py \
  --ns3-dir "$NS3" --episodes 5 --num-nodes 20 --seed 1
```

Keluarannya hanya di terminal: satu baris `target` per episode dan
satu baris `Return rata-rata`. Tidak ada file model atau CSV.

## Arti angka

State dari ns-3 berisi 44 angka: 6 slot kali 7 fitur, plus 2 fitur node.
Python membalas indeks slot `0..5`.

| Keluaran | Arti |
|---|---|
| `return=1.000 steps=1 delivered` | Paket langsung sampai. Hadiahnya +1. |
| `return=0.907 steps=2 delivered` | Sampai lewat satu relay. Ada biaya hop, lalu +1. |
| `return=-1.096 steps=2 link_loss` | Satu hop berhasil, hop berikutnya putus. Hadiah gagal -1 ditambah biaya hop. |

`Return rata-rata` adalah rata-rata return episode pada run itu.
Lima episode belum merupakan hasil pelatihan. Bobot jaringan hilang
ketika program selesai.
