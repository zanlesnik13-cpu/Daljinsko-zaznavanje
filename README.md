# Daljinsko zaznavanje — Multi-Sensor EO (Slovenija) / Remote Sensing App

Geodetsko natančna spletna aplikacija za obdelavo in analizo večsenzorskih satelitskih posnetkov (**Sentinel-2 L2A**, **Landsat Collection 2 Level-2 1984–danes** in **Sentinel-1 SAR RTC**) v slovenskem državnem koordinatnem sistemu **D96/TM (`EPSG:3794`)** in **WGS 84 / UTM 33N (`EPSG:32633`)**, zasnovana za neposredno pretočno nalaganje oblačnih rastrov (COG) preko **Microsoft Planetary Computer STAC API** ter optimizirana za delovanje znotraj omejitve **1 GB RAM** (Streamlit Community Cloud).

> **Bilingual UI:** The interface supports instant live switching between **Slovenščina (`SL`)** and **English (`EN`)** directly in the sidebar.

---

## Ključne zmogljivosti / Key Features

- **3 satelitski senzorji / 3 Multi-Sensor Archives:**
  - **Sentinel-2 L2A (`10 m` | `2015–danes`):** optični posnetki visoke ločljivosti z ESA PB `>= 04.00` kalibracijo in SCL oblačno masko.
  - **Landsat C2 L2 (`30 m` | `1984–danes`):** več kot 40-letni arhiv Landsat 4/5/7/8/9, harmoniziran na logične kanale (`B02/B03/B04/B08/B11/B12`), z uradno USGS kalibracijo odbojnosti (`DN × 0.0000275 − 0.2`), označevanjem/filtriranjem Landsat 7 SLC-off in bitno oblačno masko `QA_PIXEL`.
  - **Sentinel-1 SAR RTC (`10 m` | `Radar — skozi oblake`):** radarski posnetki C-band SAR (neodvisno od oblačnosti) s pretvorbo linearne jakosti v decibele (`10 · log10(max(val, 1e-6))`), vizualizacijo `VV [dB]`, `VH [dB]` in `SAR False Color (R: VV, G: VH, B: VV/VH)`.
- **5 načinov določitve območja zanimanja (AOI) + iskanje krajev / 5 AOI Modes + Place Search:**
  - Velik 600 px interaktivni zemljevid (klik + metrični radij v `EPSG:3794`) in iskanje po imenu kraja (**Nominatim** s prednostjo za Slovenijo)
  - **Interaktivno risanje poligona / pravokotnika** (`folium.plugins.Draw`) z natančnim izrezom rastrov (`rioxarray.rio.clip`) za vse karte, statistiko in časovne vrste
  - Ročni vnos koordinat v **D96/TM (`e, n [m]`)** ali **WGS84 (`lat, lon [°]`)** s pretvorbo v živo
  - 6 prednastavljenih lokacij v Sloveniji (*Ljubljana*, *Triglav / Bohinjsko jezero*, *Maribor / Pohorje*, *Koper / Sečoveljske soline*, *Kras (požarišče)*, *Mura / Prekmurje*)
  - Nalaganje lastnega **GeoJSON** poligona (`EPSG:4326`, z natančnim `rio.clip` izrezom)
- **Zavihek 1 — Iskanje posnetkov in pregled (RGB / NIR / SWIR / SAR):**
  - Interaktivni **Plotly zoom/pan** prikazovalnik s fiksno višino (`580 px`, merilo 1:1, sever zgoraj), kontrastni razteg (2 %–98 %), gama korekcija in izvoz večkanalnega **GeoTIFF (.tif)** za QGIS / ArcGIS.
- **Zavihek 2 — Spektralni indeksi, spektralni podpis, raba tal ali SAR maska voda / poplav:**
  - *Optika (S2 & Landsat):* **NDVI**, **NDWI**, **MNDWI**, **NDBI** in **NBR**, samodejna oblačna maska (`SCL` ali `QA_PIXEL`), **spektralni podpis točke (`490–2190 nm`)**, **hitro razvrščanje rabe tal v 4 razrede (`ha` in `%`)**, interaktivni histogram in izvoz **GeoTIFF**.
  - *Radar (Sentinel-1 SAR):* dinamična maska vodnih površin in poplav (`VV < prag [dB]`, privzeto `-16.0 dB`) z izračunom površine v `ha`, histogramom `VV [dB]` in izvozom dvoklasne maske kot **uint8 GeoTIFF**.
- **Zavihek 3 — Časovne vrste, zaznava sprememb ($\Delta = t_2 - t_1$), Prej/Potem drsnik in Timelapse:**
  - Pomnilniško varna sekvenčna časovna vrsta za vse 3 senzorje (ena epoha v RAM-u naenkrat), interaktivni Plotly graf (`Mean`, `Median`, pas `P10–P90`) z izvozom **CSV**.
  - Natančna 1-na-1 pikselska zaznava sprememb med dvema datumoma (`rioxarray.reproject_match`), **interaktivni drsnik Prej / Potem (t₁ ↔ t₂)**, analiza površin porasta/upada v `ha` in izvoz karte sprememb **GeoTIFF**.
  - **Vizualni Timelapse (GIF + drsnik po datumih):** pomnilniško varna animacija sprememb skozi čas (RGB, NIR, indeksi ali SAR VV dB) z datumskim žigom, prenosom `.gif` in ročnim pregledovanjem posameznih datumov.

---

## Geodetska in radiometrična natančnost / Geodetic & Radiometric Accuracy

| Lastnost / Property | Implementacija / Implementation |
|---|---|
| **Koordinatni sistemi (CRS)** | `EPSG:3794` (D96/TM), `EPSG:32633` (UTM 33N), `EPSG:4326` (WGS84, strogi vrstni red `(x, y)` / `(lon, lat)`) |
| **Sentinel-2 L2A kalibracija** | Samodejno zaznavanje ESA Processing Baseline `>= 04.00`: `(DN - 1000) / 10000.0`, sicer `DN / 10000.0`, omejeno na `[0.0, 1.0]` (`float32`), maska `SCL` (`0, 1, 3, 8, 9, 10`) |
| **Landsat C2 L2 kalibracija** | Uradna USGS formula `DN * 0.0000275 - 0.2`, omejeno na `[0.0, 1.0]` (`float32`), bitna maska `QA_PIXEL` (`(qa & 0b11110) != 0` in fill bit 0) |
| **Sentinel-1 SAR RTC** | Pretvorba linearnega povratnega sipa v decibele: `10 * log10(max(val, 1e-6))` (`float32`), razmerje `VV/VH [dB] = VV_dB - VH_dB` |
| **Izrez po poligonu** | Natančen rastrski izrez (`rioxarray.rio.clip`) za narisane (`folium.plugins.Draw`) in naložene GeoJSON poligone |
| **Zaščita 1 GB RAM** | Pretakanje samo izseka AOI in potrebnih kanalov, omejen GDAL predpomnilnik (`64 MB`), samodejno priporočilo ločljivosti (`10/20/30/60 m`), omejitev AOI `≤ 900 km²`, takojšnje sproščanje (`gc.collect()`) |

---

## Struktura repozitorija / Repository Structure

```text
.
├── .streamlit/
│   └── config.toml        # Kartografska svetla tema (#0F766E / #F8FAFC)
├── app.py                 # Glavna večsenzorska Streamlit aplikacija (SL / EN)
├── geodezija_crs.py       # CRS pretvorbe (D96/TM, WGS84, UTM 33N), AOI, rio.clip, GeoTIFF izvoz
├── stac_loader.py         # Planetary Computer STAC & COG nalaganje (Sentinel-2, Landsat C2 L2, Sentinel-1 RTC)
├── analize.py             # SCL/QA_PIXEL oblačne maske, optični indeksi, SAR metrike in vodna maska
├── casovne_vrste.py       # Sekvenčne časovne vrste in pikselsko poravnana zaznava sprememb za vse 3 senzorje
├── i18n.py                # Dvojezični slovar (Slovenščina / English)
├── test_step1.py          # Avtomatski verifikacijski test zaledja nad Ljubljano
└── requirements.txt       # Odvisnosti za lokalni zagon in Streamlit Cloud
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

Preverjanje geodetskega in STAC zaledja:
```bash
python test_step1.py
```

---

## Objava na Streamlit Community Cloud / Deploying to Streamlit Community Cloud

1. Potisnite repozitorij na **GitHub**.
2. Odprite [share.streamlit.io](https://share.streamlit.io/) in izberite **New app**.
3. Izberite ta repozitorij, vejo `main` in glavno datoteko **`app.py`**.
4. Kliknite **Deploy!** — aplikacija ne potrebuje API ključev (Microsoft Planetary Computer uporablja anonimno SAS podpisovanje preko `planetary-computer`).
