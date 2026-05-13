# 输出数据列表

## 数据集: PG-19 (Fiction)

Context Length: 1024

| Algorithm | TTFT mean±std (s) | TPOT mean±std (s/token) | Throughput mean±std (tk/s) | PPL mean±std |
| --- | --- | --- | --- | --- |
| 1. Baseline (Dense) | 0.0397 ± 0.0039 | 0.0057 ± 0.0003 | 175.31 ± 8.33 | 31.6089 ± 0.0000 |
| 2. StreamingLLM (Ratio=0.5, Cap=512) | 0.0482 ± 0.0045 | 0.0061 ± 0.0001 | 164.39 ± 0.97 | 91.4922 ± 0.0000 |
| 3. H2O (Ratio=0.5, Cap=512) | 0.0452 ± 0.0092 | 0.0093 ± 0.0013 | 109.37 ± 16.26 | 93.2483 ± 0.0000 |
| 4. SnapKV (Ratio=0.5, Cap=512) | 0.0466 ± 0.0085 | 0.0084 ± 0.0004 | 119.92 ± 6.40 | 83.1132 ± 0.0000 |

Context Length: 2048

| Algorithm | TTFT mean±std (s) | TPOT mean±std (s/token) | Throughput mean±std (tk/s) | PPL mean±std |
| --- | --- | --- | --- | --- |
| 1. Baseline (Dense) | 0.0858 ± 0.0038 | 0.0199 ± 0.0003 | 50.34 ± 0.88 | 30.4283 ± 0.0000 |
| 2. StreamingLLM (Ratio=0.5, Cap=1024) | 0.1107 ± 0.0203 | 0.0063 ± 0.0011 | 161.54 ± 25.64 | 94.1676 ± 0.0000 |
| 3. H2O (Ratio=0.5, Cap=1024) | 0.1195 ± 0.0101 | 0.0085 ± 0.0020 | 120.45 ± 24.39 | 92.6860 ± 0.0000 |
| 4. SnapKV (Ratio=0.5, Cap=1024) | 0.1118 ± 0.0193 | 0.0087 ± 0.0013 | 115.82 ± 15.99 | 88.3608 ± 0.0000 |

Context Length: 3072

| Algorithm | TTFT mean±std (s) | TPOT mean±std (s/token) | Throughput mean±std (tk/s) | PPL mean±std |
| --- | --- | --- | --- | --- |
| 1. Baseline (Dense) | 0.4752 ± 0.0022 | 0.0567 ± 0.0010 | 17.64 ± 0.30 | 32.0812 ± 0.0000 |
| 2. StreamingLLM (Ratio=0.5, Cap=1536) | 0.2974 ± 0.0127 | 0.0066 ± 0.0002 | 152.85 ± 3.55 | 91.1208 ± 0.0000 |
| 3. H2O (Ratio=0.5, Cap=1536) | 0.3115 ± 0.0180 | 0.0110 ± 0.0018 | 92.41 ± 15.66 | 96.7746 ± 0.0000 |
| 4. SnapKV (Ratio=0.5, Cap=1536) | 0.2541 ± 0.0470 | 0.0097 ± 0.0012 | 104.38 ± 13.91 | 84.3961 ± 0.0000 |

Context Length: 4096

| Algorithm | TTFT mean±std (s) | TPOT mean±std (s/token) | Throughput mean±std (tk/s) | PPL mean±std |
| --- | --- | --- | --- | --- |
| 1. Baseline (Dense) | 0.6727 ± 0.0230 | 0.0957 ± 0.0093 | 10.51 ± 0.96 | 64.8814 ± 0.0000 |
| 2. StreamingLLM (Ratio=0.5, Cap=2048) | 0.6973 ± 0.0359 | 0.0072 ± 0.0010 | 140.73 ± 17.29 | 89.8794 ± 0.0000 |
| 3. H2O (Ratio=0.5, Cap=2048) | 0.6608 ± 0.0259 | 0.0164 ± 0.0002 | 61.05 ± 0.81 | 90.5961 ± 0.0000 |
| 4. SnapKV (Ratio=0.5, Cap=2048) | 0.7243 ± 0.0511 | 0.0134 ± 0.0009 | 75.18 ± 4.81 | 85.3144 ± 0.0000 |

## 数据集: Wikitext (Encyclopedia)

Context Length: 1024

| Algorithm | TTFT mean±std (s) | TPOT mean±std (s/token) | Throughput mean±std (tk/s) | PPL mean±std |
| --- | --- | --- | --- | --- |
| 1. Baseline (Dense) | 0.0521 ± 0.0033 | 0.0063 ± 0.0004 | 159.31 ± 8.64 | 25.4820 ± 0.0000 |
| 2. StreamingLLM (Ratio=0.5, Cap=512) | 0.0513 ± 0.0105 | 0.0067 ± 0.0008 | 149.11 ± 15.11 | 110.8550 ± 0.0000 |
| 3. H2O (Ratio=0.5, Cap=512) | 0.0541 ± 0.0038 | 0.0078 ± 0.0002 | 127.77 ± 3.31 | 109.4308 ± 0.0000 |
| 4. SnapKV (Ratio=0.5, Cap=512) | 0.0540 ± 0.0023 | 0.0084 ± 0.0016 | 122.25 ± 21.67 | 80.4543 ± 0.0000 |

Context Length: 2048

| Algorithm | TTFT mean±std (s) | TPOT mean±std (s/token) | Throughput mean±std (tk/s) | PPL mean±std |
| --- | --- | --- | --- | --- |
| 1. Baseline (Dense) | 0.1151 ± 0.0189 | 0.0186 ± 0.0034 | 55.22 ± 10.58 | 31.5087 ± 0.0000 |
| 2. StreamingLLM (Ratio=0.5, Cap=1024) | 0.1197 ± 0.0011 | 0.0069 ± 0.0003 | 145.89 ± 6.03 | 107.4906 ± 0.0000 |
| 3. H2O (Ratio=0.5, Cap=1024) | 0.1165 ± 0.0082 | 0.0100 ± 0.0013 | 100.91 ± 12.57 | 109.5712 ± 0.0000 |
| 4. SnapKV (Ratio=0.5, Cap=1024) | 0.1012 ± 0.0184 | 0.0088 ± 0.0013 | 115.75 ± 18.27 | 84.0893 ± 0.0000 |

Context Length: 3072

| Algorithm | TTFT mean±std (s) | TPOT mean±std (s/token) | Throughput mean±std (tk/s) | PPL mean±std |
| --- | --- | --- | --- | --- |
| 1. Baseline (Dense) | 0.5126 ± 0.0508 | 0.0527 ± 0.0029 | 19.04 ± 1.09 | 33.8519 ± 0.0000 |
| 2. StreamingLLM (Ratio=0.5, Cap=1536) | 0.2679 ± 0.0459 | 0.0059 ± 0.0004 | 169.51 ± 11.58 | 98.1664 ± 0.0000 |
| 3. H2O (Ratio=0.5, Cap=1536) | 0.3094 ± 0.0065 | 0.0089 ± 0.0004 | 112.25 ± 5.19 | 98.6320 ± 0.0000 |
| 4. SnapKV (Ratio=0.5, Cap=1536) | 0.2430 ± 0.0253 | 0.0075 ± 0.0006 | 133.41 ± 10.56 | 81.9234 ± 0.0000 |

Context Length: 4096

| Algorithm | TTFT mean±std (s) | TPOT mean±std (s/token) | Throughput mean±std (tk/s) | PPL mean±std |
| --- | --- | --- | --- | --- |
| 1. Baseline (Dense) | 0.6470 ± 0.0908 | 0.0912 ± 0.0009 | 10.97 ± 0.11 | 79.3191 ± 0.0000 |
| 2. StreamingLLM (Ratio=0.5, Cap=2048) | 0.3765 ± 0.0166 | 0.0081 ± 0.0008 | 123.69 ± 11.49 | 103.5440 ± 0.0000 |
| 3. H2O (Ratio=0.5, Cap=2048) | 0.6464 ± 0.0922 | 0.0145 ± 0.0016 | 69.44 ± 7.16 | 102.0167 ± 0.0000 |
| 4. SnapKV (Ratio=0.5, Cap=2048) | 0.6207 ± 0.0753 | 0.0107 ± 0.0008 | 93.74 ± 6.67 | 90.2562 ± 0.0000 |
