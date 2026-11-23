---
title: Daljinsko Zaznavanje Multi-Sensor EO
emoji: 🛰️
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.50.0"
app_file: app.py
pinned: false
---

# Daljinsko zaznavanje — Multi-Sensor EO (Slovenija) / Remote Sensing App

Geodetsko natančna spletna aplikacija za obdelavo in analizo večsenzorskih satelitskih posnetkov (**Sentinel-2 L2A**, **Landsat Collection 2 Level-2 1984–danes** in **Sentinel-1 SAR RTC**) v slovenskem državnem koordinatnem sistemu **D96/TM (`EPSG:3794`)** ter samodejnem lokalnem **WGS 84 / UTM** po vsem svetu, zasnovana za neposredno pretočno nalaganje oblačnih rastrov (COG) preko **Microsoft Planetary Computer STAC API** ter optimizirana za **Hugging Face Spaces (16 GB RAM, 2 vCPU)** s polno **10 m** ločljivostjo Sentinel in 4-nitnim vzporednim nalaganjem.

> **Bilingual UI:** The interface supports instant live switching between **Slovenščina (`SL`)** and **English (`EN`)** directly in the sidebar.

---

## Ključne zmogljivosti / Key Features

- **3 satelitski senzorji / 3 Multi-Sensor Archives:**
  - **Sentinel-2 L2A (`10 m` | `2015–danes`):** optični posnetki polne ločljivosti 10 m (do 1200 km² AOI) z ESA PB `>= 04.00` kalibracijo in SCL oblačno masko.
  - **Landsat C2 L2 (`30 m` | `1984–danes`):** več kot 40-letni arhiv Landsat 4/5/7/8/9, harmoniziran na logične kanale (`B02/B03/B04/B08/B11/B12/LST`), z uradno USGS kalibracijo odbojnosti (`DN × 0.0000275 − 0.2`), temperaturo površja (`LST [°C]`), izločanjem Landsat 7 SLC-off in bitno oblačno masko `QA_PIXEL`.
  - **Sentinel-1 SAR RTC (`10 m` | `Radar — skozi oblake`):** radarski posnetki C-band SAR (neodvisno od oblačnosti) s pretvorbo linearne jakosti v decibele (`10 · log10(max(val, 1e-6))`), vizualizacijo `VV [dB]`, `VH [dB]` in `SAR False Color (R: VV, G: VH, B: VV/VH)`.
- **5 načinov določitve območja zanimanja (AOI) + iskanje krajev / 5 AOI Modes + Place Search:**
  - Velik 600 px interaktivni zemljevid (klik + metrični radij v `EPSG:3794` / lokalnem UTM) in predpomnjeno iskanje po imenu kraja (**Photon / Nominatim** s prednostjo za Slovenijo)
  - **Interaktivno risanje poligona / pravokotnika** (`folium.plugins.Draw`) z natančnim izrezom rastrov (`rioxarray.rio.clip`) za vse karte, statistiko in časovne vrste
  - Ročni vnos koordinat v **D96/TM (`e, n [m]`)** ali **WGS84 (`lat, lon [°]`)** s pretvorbo v živo
  - 6 prednastavljenih lokacij v Sloveniji (*Ljubljana*, *Triglav / Bohinjsko jezero*, *Maribor / Pohorje*, *Koper / Sečoveljske soline*, *Kras (požarišče)*, *Mura / Prekmurje*)
  - Nalaganje lastnega **GeoJSON** poligona (`EPSG:4326`, z natančnim `rio.clip` izrezom)
- **Zavihek 1 — Iskanje posnetkov in pregled (RGB / NIR / SWIR / SAR):**
  - Visokoločljivostni interaktivni slippy-map prikazovalnik (do **1800 px** PNG prekrivalo), kontrastni razteg (2 %–98 %), gama korekcija in izvoz večkanalnega **GeoTIFF (.tif)** za QGIS / ArcGIS.
- **Zavihek 2 — Spektralni indeksi, relief (Copernicus DEM 30 m), sneg/led (NDSI), LST (°C), raba tal ali SAR maska poplav:**
  - *Optika (S2 & Landsat):* **NDVI**, **NDWI**, **MNDWI**, **NDBI**, **NBR**, **NDSI** (snežna meja z DEM), **LST (°C)**, samodejna oblačna maska (`SCL` ali `QA_PIXEL`), **spektralni podpis točke (`490–2190 nm`)**, **hitro razvrščanje rabe tal v 4 razrede (`ha` in `%`)**, **brezoblačni mediani mozaik** in izvoz **GeoTIFF**.
  - *Radar (Sentinel-1 SAR):* dinamična maska vodnih površin in poplav (`VV < prag [dB]`, privzeto `-16.0 dB`) z izračunom površine v `ha`, histogramom `VV [dB]` in izvozom dvoklasne maske kot **uint8 GeoTIFF**.
- **Zavihek 3 — Časovne vrste, zaznava sprememb ($\Delta = t_2 - t_1$), Prej/Potem drsnik in Timelapse:**
  - **4-nitno vzporedno pretakanje (`ThreadPoolExecutor(max_workers=4)`)** za časovne vrste, vizualni Timelapse (GIF) in brezoblačne mediane mozaike (3–4× hitrejša obdelava s sprotnim `gc.collect()`).
  - Natančna 1-na-1 pikselska zaznava sprememb med dvema datumoma (`rioxarray.reproject_match`), **interaktivni drsnik Prej / Potem (t₁ ↔ t₂)**, analiza površin porasta/upada v `ha` in izvoz karte sprememb **GeoTIFF**.

---

## Geodetska in radiometrična natančnost / Geodetic & Radiometric Accuracy

| Lastnost / Property | Implementacija / Implementation |
|---|---|
| **Koordinatni sistemi (CRS)** | `EPSG:3794` (D96/TM v Sloveniji), samodejni globalni **WGS 84 / UTM** (`EPSG:326xx` / `327xx`), `EPSG:4326` (WGS84, strogi vrstni red `(x, y)` / `(lon, lat)`) |
| **Sentinel-2 L2A kalibracija** | Samodejno zaznavanje ESA Processing Baseline `>= 04.00`: `(DN - 1000) / 10000.0`, sicer `DN / 10000.0`, omejeno na `[0.0, 1.0]` (`float32`), maska `SCL` (`0, 1, 3, 8, 9, 10`) |
| **Landsat C2 L2 kalibracija** | Uradna USGS formula `DN * 0.0000275 - 0.2`, termalna `LST = DN * 0.00341802 + 149.0 - 273.15 [°C]`, bitna maska `QA_PIXEL` (`(qa & 0b11110) != 0` in fill bit 0) |
| **Sentinel-1 SAR RTC** | Pretvorba linearnega povratnega sipa v decibele: `10 * log10(max(val, 1e-6))` (`float32`), razmerje `VV/VH [dB] = VV_dB - VH_dB` |
| **Izrez po poligonu** | Natančen rastrski izrez (`rioxarray.rio.clip`) za narisane (`folium.plugins.Draw`) in naložene GeoJSON poligone |
| **Profil 16 GB RAM (HF Spaces)** | Polna **10 m** ločljivost Sentinel do **1200 km²** (`20 m` nad 1200 km², `30 m` nad 2500 km², omejitev AOI `≤ 4000 km²`), 1800 px prekrivala, `GDAL_CACHEMAX=256 MB`, 4-nitno vzporedno nalaganje in `@st.cache_data` (1 h) |

---

## Struktura repozitorija / Repository Structure

```text
.
├── .streamlit/
│   └── config.toml        # Kartografska svetla tema (#0F766E / #F8FAFC)
├── app.py                 # Glavna večsenzorska Streamlit aplikacija (SL / EN)
├── geodezija_crs.py       # CRS pretvorbe (D96/TM, globalni auto-UTM, WGS84), AOI, rio.clip, GeoTIFF izvoz
├── stac_loader.py         # Planetary Computer STAC & COG nalaganje s predpomnjenjem (S2, Landsat, S1, DEM)
├── analize.py             # SCL/QA_PIXEL oblačne maske, optični indeksi, LST, NDSI, SAR metrike in vodna maska
├── casovne_vrste.py       # 4-nitne vzporedne časovne vrste, Timelapse, mediani mozaiki in zaznava sprememb
├── relief.py              # Copernicus DEM 30 m analiza terena (naklon, ekspozicija, senčenje, profil A→B)
├── map_overlay.py         # 1800 px RGBA prekrivala za interaktivne Folium slippy-map zemljevide
├── ai_pomocnik.py         # AI pomočnik (Gemini REST API preko HF/Streamlit Secrets + lokalni parser)
├── gnss_plan.py           # GNSS načrtovanje meritev (orbite + DEM obzorje + S2 ovire)
├── nesrece.py             # Hitra analiza naravnih nesreč (poplave SAR, požari dNBR, vetrolom ΔNDVI)
├── kartografija.py        # Kartografski izvoz (tiskalne karte z mrežo, merilom in legendo)
├── i18n.py                # Dvojezični slovar (Slovenščina / English)
├── packages.txt           # Sistemski Debian paketi (GDAL/PROJ/GEOS/pisave) za Hugging Face Spaces
└── requirements.txt       # Python odvisnosti
```

---

## Lokalni zagon / Running Locally

Zahtevan je **Python 3.11+**.

```bash
python -m venv .venv

# Linux / macOS:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
streamlit run app.py
```

---

## Objava na Hugging Face Spaces / Deploying to Hugging Face Spaces

1. Ustvarite nov Space na [huggingface.co/spaces](https://huggingface.co/spaces) z izbiro **SDK: Streamlit** (brezplačni **CPU basic: 2 vCPU, 16 GB RAM**).
2. Povežite ali potisnite ta repozitorij — `README.md` že vsebuje veljavno YAML glavo (`sdk: streamlit`, `app_file: app.py`) in `packages.txt` poskrbi za sistemske knjižnice.
3. *(Opcijsko)* V **Space Settings → Variables and secrets → New secret** dodajte `GEMINI_API_KEY` za uporabo AI pomočnika z Gemini API (aplikacija samodejno bere `os.environ["GEMINI_API_KEY"]` in deluje tudi brez ključa z vgrajenim lokalnim parserjem).
