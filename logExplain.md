# Penjelasan log TLDQN Gym

Log di bawah ini dari lima episode `python ml/tldqn_routing_gym.py` terhadap `honeycomb-sdwsn-gym`, 20 sensor, seed awal 1.

```text
Latih TLDQN online, 5 episode, 20 sensor ...
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
Waiting for simulation script to connect on port: tcp://localhost:5555
Please start proper ns-3 simulation script using ./waf --run "..."
[0/2] Re-checking globbed directories...
ninja: no work to do.
honeycomb-sdwsn-gym: 20 sensor, port 5555, satu episode paket
Simulation process id: 10716 (parent (waf shell) id: 10708)
Waiting for Python process to connect on port: tcp://localhost:5555
Please start proper Python Gym Agent
  target 1: return=1.000 steps=1 delivered
Waiting for simulation script to connect on port: tcp://localhost:5555
Please start proper ns-3 simulation script using ./waf --run "..."
[0/2] Re-checking globbed directories...
ninja: no work to do.
honeycomb-sdwsn-gym: 20 sensor, port 5555, satu episode paket
Simulation process id: 10736 (parent (waf shell) id: 10728)
Waiting for Python process to connect on port: tcp://localhost:5555
Please start proper Python Gym Agent
  target 2: return=-1.096 steps=2 link_loss
Waiting for simulation script to connect on port: tcp://localhost:5555
Please start proper ns-3 simulation script using ./waf --run "..."
[0/2] Re-checking globbed directories...
ninja: no work to do.
honeycomb-sdwsn-gym: 20 sensor, port 5555, satu episode paket
Simulation process id: 10747 (parent (waf shell) id: 10739)
Waiting for Python process to connect on port: tcp://localhost:5555
Please start proper Python Gym Agent
  target 3: return=1.000 steps=1 delivered
Waiting for simulation script to connect on port: tcp://localhost:5555
Please start proper ns-3 simulation script using ./waf --run "..."
[0/2] Re-checking globbed directories...
ninja: no work to do.
honeycomb-sdwsn-gym: 20 sensor, port 5555, satu episode paket
Simulation process id: 10758 (parent (waf shell) id: 10750)
Waiting for Python process to connect on port: tcp://localhost:5555
Please start proper Python Gym Agent
  target 4: return=0.907 steps=2 delivered
Waiting for simulation script to connect on port: tcp://localhost:5555
Please start proper ns-3 simulation script using ./waf --run "..."
[0/2] Re-checking globbed directories...
ninja: no work to do.
honeycomb-sdwsn-gym: 20 sensor, port 5555, satu episode paket
Simulation process id: 10769 (parent (waf shell) id: 10761)
Waiting for Python process to connect on port: tcp://localhost:5555
Please start proper Python Gym Agent
  target 5: return=1.000 steps=1 delivered
Return rata-rata: 0.562
```

## Apa yang sedang terjadi

Lima baris `target` adalah lima percobaan kirim paket. Satu episode sama dengan satu proses ns-3 dan satu paket, dari satu sensor menuju sink terdekat. Setelah paket selesai, proses ns-3 ditutup, lalu episode berikutnya membuka proses baru di port yang sama.

Peringatan Gym di awal hanya memberitahu bahwa pustaka `gym` lama sudah tidak dirawat. Sambungan Python–ns-3 tetap jalan. Kalimat `Please start proper ns-3 simulation script using ./waf` adalah pesan bawaan `ns3gym` selagi Python menunggu di port 5555. Skrip Python-nya sendiri yang kemudian menjalankan ns-3, jadi perintah `waf` itu tidak perlu dijalankan manual. `ninja: no work to do.` berarti program `honeycomb-sdwsn-gym` sudah terkompilasi.

Satu episode berlangsung seperti ini. Python menunggu di `tcp://localhost:5555`. ns-3 memasang 20 sensor dan mengirim keadaan 44 angka. Agen memilih satu dari enam tetangga. ns-3 mengembalikan reward. Itu diulang sampai paket sampai, hilang, atau gagal.

- `steps` adalah jumlah pilihan tetangga pada episode itu.
- `return` adalah jumlah reward episode itu.
- Kata di ujung baris (`delivered`, `link_loss`) adalah hasil terakhir paket.

Seed bergeser tiap episode (`seed + nomor episode`), jadi posisi node dan undian kanalnya berbeda. Epsilon bawaan 0.2 berarti sekitar satu dari lima pilihan masih acak.

## Arti tiap episode

Reward di `honeycomb-sdwsn-gym.cc` mengikuti kontrak yang sama dengan lingkungan sintetis:

- paket sampai di sink: +1
- hilang di kanal, di-drop relay, TTL habis, atau aksi tidak sah: −1
- hop ke relay yang bukan sink: −0.05 − 0.10 × (1 − energi kandidat)

| Episode | Return | Langkah | Arti |
|---|---|---|---|
| 1, 3, 5 | 1.000 | 1 | Tetangga yang dipilih adalah sink. Paket langsung sampai, reward +1, tanpa biaya hop. |
| 4 | 0.907 | 2 | Hop pertama berhasil ke relay (biaya sekitar −0.093), hop kedua sampai di sink (+1). |
| 2 | −1.096 | 2 | Hop pertama berhasil ke relay (biaya sekitar −0.096), hop kedua gagal karena undian kanal (`link_loss`, reward −1). |

Rata-rata dihitung dari kelima return:

(1.000 − 1.096 + 1.000 + 0.907 + 1.000) / 5 = **0.562**

## Batas log ini

Lima episode ini membuktikan Python dan ns-3 saling membalas satu paket. Bobot jaringan tidak disimpan, dan tidak ada CSV metrik. Angka 0.562 bukan throughput, PDR, atau akurasi deteksi pada paper.
