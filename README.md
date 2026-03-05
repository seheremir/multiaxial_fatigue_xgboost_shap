# Çok Eksenli Yorulma Ömrü Tahmini: XGBoost & SHAP Analizi

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-FF6600?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-0.42%2B-00A86B?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
![Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)
![Status](https://img.shields.io/badge/Status-Under%20Review-orange?style=flat-square)

**Makale:** *Metalik Malzemelerde Çok Eksenli Yorulma Ömrünün Tahmininde XGBoost ve SHAP Analizi: Yükleme Yolu Kategorilerine Göre Malzeme Parametrelerinin Göreli Önemi*

[Yazar Adı Soyadı] · [Bölüm, Üniversite] · 2025

[📄 Makale](#makale) · [🚀 Hızlı Başlangıç](#hızlı-başlangıç) · [📊 Sonuçlar](#sonuçlar) · [📁 Repo Yapısı](#repo-yapısı)

</div>

---

## Genel Bakış

Bu çalışma, 40 metalik malzemeye ait **1.167 çok eksenli yorulma** örneği üzerinde:

- Klasik SWT (Smith-Watson-Topper) kriterini **XGBoost** ile karşılaştırır
- **SHAP** analizi ile malzeme parametrelerinin yorulma ömrüne katkısını ölçer  
- Tek eksenli, orantılı ve orantısız yükleme grupları arasında özellik öneminin farklılık gösterip göstermediğini **Bonferroni düzeltmeli Mann-Whitney U** testiyle istatistiksel olarak sınar

| Model | R² | RMSE | ±3x Kapsam |
|-------|----|------|------------|
| Ridge (Temel) | 0.119 | 0.837 | 47.3% |
| Rastgele Orman | 0.847 | 0.349 | 84.4% |
| **XGBoost (Bu Çalışma)** | **0.856** | **0.339** | **83.8%** |
| SWT (Klasik) | 0.070 | 0.848 | 42.0% |

---

## Veri Seti

> Chen, S., Bai, Y., Zhou, X. & Yang, A. *A deep learning dataset for metal multiaxial fatigue life prediction.* **Scientific Data** 11, 1027 (2024). [https://doi.org/10.1038/s41597-024-03862-4](https://doi.org/10.1038/s41597-024-03862-4)

- 40 metalik malzeme (çelik, alüminyum, paslanmaz çelik, magnezyum, titanyum, bakır, nikel)  
- 48 farklı yükleme yolu  
- 1.167 örnek: 826 tek eksenli / 191 orantılı / 150 orantısız  
- 4 malzeme özelliği: E (GPa), σ_y (MPa), σ_u (MPa), ν  

**Veri seti bu repoda bulunmamaktadır.** Yukarıdaki DOI bağlantısından erişebilirsiniz.

---

## Metodoloji

```
Ham Zaman Serisi (eksenel + kayma)
        ↓
Özellik Çıkarımı (σ_a, τ_a, λ, R, phase_proxy, σ_mean)
        ↓
Malzeme Özellikleri + LP Kodlaması  →  10 boyutlu girdi vektörü
        ↓
Stratified Train/Test Split (%80 / %20)
        ↓
XGBoost Eğitimi  →  5×10 Tekrarlı K-Fold CV
        ↓
SHAP TreeExplainer (sadece test seti)
        ↓
Bootstrap CI (n_boot=1000) + Mann-Whitney U (Bonferroni)
```

### Metodolojik Seçimler

| Konu | Uygulama | Gerekçe |
|------|----------|---------|
| Train/test bölümü | LP kategorisine göre **stratified** | Dengesiz sınıf dağılımı (826/191/150) |
| SHAP hesabı | Yalnızca **test setine** | Eğitim verisinden bağımsız tarafsız ölçüm |
| Bootstrap CI | Test alt kümelerine, n=1000 | Grup bazlı güven aralıkları |
| İstatistiksel test | Bonferroni düzeltmeli Mann-Whitney U | Çoklu karşılaştırma kontrolü (α=0.05/12) |
| Kararlılık | 5×10 tekrarlı K-Fold | Tek split varyansını kontrol |
| SWT karşılaştırması | Kalibre edilmemiş formülasyon | 40 malzeme için evrensel baseline |

---

## Sonuçlar

### Global SHAP Özellik Önemi

| Sıra | Özellik | Açıklama | Ort. \|SHAP\| |
|------|---------|----------|--------------|
| 1 | σ_a | Eksenel gerilme genliği | 0.391 |
| 2 | τ_a | Kayma gerilmesi genliği | 0.249 |
| 3 | phase_proxy | Faz açısı vekili | 0.086 |
| 4 | σ_y | Akma dayanımı | 0.080 |
| 5 | E | Elastisite modülü | 0.077 |
| 6 | σ_u | Nihai çekme dayanımı | 0.068 |
| 7 | ν | Poisson oranı | 0.066 |
| 8 | λ | Gerilme oranı τ_a/σ_a | 0.036 |
| 9 | lp_encoded | Yükleme yolu kategorisi | 0.026 |
| 10 | R | Yük oranı | 0.001 |

### Ana Bulgular

1. **XGBoost**, SWT kriterini belirgin biçimde geride bırakmıştır (R²: 0.856 vs 0.070)
2. **σ_a** tüm yükleme kategorilerinde baskın tahmin edici konumundadır
3. Malzeme parametrelerinin göreli önemi **yükleme yolu türlerine göre istatistiksel olarak anlamlı farklılık göstermemektedir** (Bonferroni düzeltmeli Mann-Whitney, tüm p > 0.0042)
4. **lp_encoded** en düşük SHAP önemine sahiptir — yükleme yolu etkisi, σ_a, τ_a ve phase_proxy aracılığıyla zaten modele aktarılmaktadır

---

## Hızlı Başlangıç

### Google Colab (Önerilen)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KULLANICI_ADI/REPO_ADI/blob/main/multiaxial_fatigue_xgboost_shap.ipynb)

1. Yukarıdaki butona tıklayın
2. Veri setini indirip Drive'a yükleyin
3. `HÜCRE 1`'deki yol değişkenlerini güncelleyin
4. `Çalışma zamanı → Tümünü çalıştır`

### Yerel Kurulum

```bash
# Repoyu klonlayın
git clone https://github.com/KULLANICI_ADI/REPO_ADI.git
cd REPO_ADI

# Sanal ortam oluşturun (önerilen)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt

# Jupyter'ı başlatın
jupyter notebook multiaxial_fatigue_xgboost_shap.ipynb
```

---

## Repo Yapısı

```
📦 REPO_ADI
├── 📓 multiaxial_fatigue_xgboost_shap.ipynb   # Ana analiz notebook'u
├── 📋 requirements.txt                         # Python bağımlılıkları
├── 📄 README.md                                # Bu dosya
├── 📄 LICENSE                                  # MIT Lisansı
├── 📄 CITATION.cff                             # Atıf bilgisi
├── 📁 figures/                                 # Makale görselleri
│   ├── 01_veri_ozet.png
│   ├── 03_scatter_modeller.png
│   ├── 06_shap_global.png
│   ├── 07_shap_lp_karsilastirma.png
│   ├── 09_shap_interaksiyon.png
│   ├── 10_shap_mat_karsilastirma.png
│   ├── 13_swt_vs_xgboost.png
│   └── 14_shap_lp_bootstrap_ci.png
└── 📁 .github/
    └── 📁 workflows/
        └── ci.yml                              # Otomatik test (opsiyonel)
```

---

## Makale

> [Yazar Adı Soyadı]. *Metalik Malzemelerde Çok Eksenli Yorulma Ömrünün Tahmininde XGBoost ve SHAP Analizi: Yükleme Yolu Kategorilerine Göre Malzeme Parametrelerinin Göreli Önemi.* [Dergi Adı], 2025. *(Hakemlik sürecinde)*

---

## Atıf

Bu kodu kullanıyorsanız lütfen şu şekilde atıfta bulunun:

```bibtex
@article{YazarSoyadi2025,
  author  = {Yazar Adı Soyadı},
  title   = {Metalik Malzemelerde Çok Eksenli Yorulma Ömrünün Tahmininde
             XGBoost ve SHAP Analizi},
  journal = {Dergi Adı},
  year    = {2025},
  note    = {Under review}
}
```

Kullandığımız veri setine de atıfta bulunmayı unutmayın:

```bibtex
@article{Chen2024,
  author  = {Chen, S. and Bai, Y. and Zhou, X. and Yang, A.},
  title   = {A deep learning dataset for metal multiaxial fatigue life prediction},
  journal = {Scientific Data},
  volume  = {11},
  pages   = {1027},
  year    = {2024},
  doi     = {10.1038/s41597-024-03862-4}
}
```

---

## Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.

---

<div align="center">
<sub>Sorularınız için <a href="mailto:ornek@universite.edu.tr">ornek@universite.edu.tr</a> adresine yazabilirsiniz.</sub>
</div>
