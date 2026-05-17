# Kredit Təsdiqi Proqnozu Layihəsi — Sənədləşmə (Documentation)

Bu sənəd, maşın öyrənməsi metodlarından istifadə edərək bank mühitində kredit müraciətlərinin avtomatlaşdırılmış qiymətləndirilməsi layihəsinin strukturunu, data hazırlığını, modelləşdirmə addımlarını və əldə olunmuş nəticələri əks etdirir.

---

## 1. Giriş və Biznes Problemi (Problem Statement)
Maliyyə sektorunda kredit müraciətlərinin yoxlanılması çox vaxt aparan və insan subyektivliyinə açıq bir prosesdir. Bu layihədə əsas hədəf, müraciət edən şəxslərin maliyyə profilinə və keçmiş kredit tarixçəsinə əsasən, müraciətin təsdiq (`Approved`) və ya imtina (`Rejected`) alacağını proqnozlaşdıran binary classification (ikiqat təsnifat) modeli qurmaqdır.

### Biznes Üstünlükləri:
* **Sürət:** Kredit qərarlarının verilməsi müddətini saniyələrə endirir.
* **Obyektivlik:** İnsan faktorundan qaynaqlanan subyektiv səhvləri və riskləri minimuma endirir.
* **Risklərin İdarə Edilməsi:** Vaxtında ödənməyəcək riskli kreditləri əvvəlcədən təyin edərək bankın maliyyə itkilərini sığortalayır.

---

## 2. Data Setin Strukturu (Dataset Overview)
Layihədə ümumilikdə **4,269 müştəri müraciəti** və hər bir müraciəti təsvir edən **13 fərqli sütun (feature)** analiz edilmişdir. Datada heç bir itirilmiş və ya boş məlumat (`missing value`) yoxdur.

### Dəyişənlərin Təsviri:
| Sütun Adı | Tip | Təsviri |
| :--- | :--- | :--- |
| `loan_id` | Rəqəmsal | Müştərinin unikal ID nömrəsi (Model üçün faydasız olduğundan çıxarılıb) |
| `no_of_dependents` | Rəqəmsal | Himayədə olan ailə üzvlərinin sayısı |
| `education` | Kateqoriyal | Müştərinin təhsil səviyyəsi (`Graduate` / `Not Graduate`) |
| `self_employed` | Kateqoriyal | Fərdi sahibkar olub-olmaması (`Yes` / `No`) |
| `income_annum` | Rəqəmsal | İllik ümumi gəlir |
| `loan_amount` | Rəqəmsal | İstənilən kredit məbləği |
| `loan_term` | Rəqəmsal | Kreditin geri ödəniş müddəti (aylarla) |
| `cibil_score` | Rəqəmsal | Müştərinin rəsmi kredit balı (300 - 900 arası) |
| `residential_assets_value` | Rəqəmsal | Yaşayış mülkünün dəyəri |
| `commercial_assets_value` | Rəqəmsal | Kommersiya mülkünün dəyəri |
| `luxury_assets_value` | Rəqəmsal | Lüks aktivlərin dəyəri (avtomobil, qızıl və s.) |
| `bank_asset_value` | Rəqəmsal | Bank hesablarındakı nağd/depozit vəsaitlər |
| `loan_status` | Kateqoriyal | **Hədəf Dəyişən** — Kreditin vəziyyəti (`Approved` / `Rejected`) |

### Hədəf Dəyişənin Balansı (Class Balance):
* **Approved (Təsdiq):** ~62.2% (2,656 müraciət)
* **Rejected (İmtina):** ~37.8% (1,613 müraciət)
Data balanslı olduğu üçün model hər iki sinfi bərabər dərəcədə öyrənə bilmişdir.

---

## 3. Datanın Ön Emalı və Feature Engineering
Modelin datadan maksimum siqnal tuta bilməsi üçün aşağıdakı hazırlıq mərhələləri icra olunmuşdur:
1. **Mətn Təmizliyi:** Sütun adlarında və dəyişənlərin daxilində olan struktur kənar boşluqlar `strip()` funksiyası ilə tamamilə silinmişdir.
2. **Sütun Filterlənməsi:** `loan_id` dəyişəni modelin riyazi hesablamalarından kənarlaşdırılmışdır.
3. **Label Encoding:** Kateqoriyal dəyişənlər modelin oxuya biləcəyi binary (0 və 1) formata gətirilmişdir:
   * `loan_status`: Approved $ightarrow$ 1, Rejected $ightarrow$ 0
   * `education`: Graduate $ightarrow$ 1, Not Graduate $ightarrow$ 0
   * `self_employed`: Yes $ightarrow$ 1, No $ightarrow$ 0
4. **Yeni Parametrlərin Hasil Edilməsi (Feature Engineering):** Bankçılıq məntiqinə uyğun olaraq 2 yeni sütun hesablanmışdır:
   * `total_assets` (Ümumi Aktivlər) = Bütün aktiv dəyərlərinin cəmi.
   * `loan_to_income` (Kreditin Gəlirə Nisbəti) = `loan_amount` / `income_annum`

---

## 4. Modelləşdirmə və Alqoritmlərin Qarşılaşdırılması
Data **80% Təlim (Train)** və **20% Test** hissələrinə bölünərək 3 fərqli təsnifat alqoritmi sınaqdan keçirilmişdir:

| Alqoritm | Train Accuracy | Test Accuracy | ROC-AUC Score | 5-Fold CV Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | ~90.2% | ~90.1% | ~0.920 | ~90.0% |
| **Decision Tree** | 100.0% | ~97.5% | ~0.970 | ~97.2% |
| **Random Forest Classifier** | **100.0%** | **99.8%** | **0.999** | **~98.8%** |

*Nəticə:* **Random Forest Classifier** digər iki alqoritmi üstələyərək demək olar ki, sıfır xəta ilə ən yaxşı nəticəni vermişdir. 5-Fold Cross-Validation (Çarpaz Yoxlama) nəticəsinin sabit qalması modelin datanı əzbərləmədiyini (`overfitting` olmadığını) təsdiqləyir.

---

## 5. Ən Vacib Faktorlar (Feature Importance)
Random Forest modelinin qərarlarına təsir edən parametrlərin çəki dərəcələri analiz edildikdə aşağıdakı iyerarxiya müəyyən olunmuşdur:
1. **CIBIL Score (~38%):** Kredit reytinqi təkbaşına modelin qərarlarının böyük bir hissəsini idarə edir. Kredit balı aşağı olan müştərilərin müraciəti birbaşa rədd edilir.
2. **Income & Loan Amount (~26%):** İllik gəlir və tələb olunan kredit məbləğinin balansı müştərinin ödəniş potensialını müəyyənləşdirir.
3. **Engineered Features (Aktivlər və Nisbətlər):** Feature engineering mərhələsində bizim yaratdığımız `total_assets` və `loan_to_income` parametrləri modelin dəqiqliyinə ciddi şəkildə müsbət təsir göstərmişdir.

---

## 6. Yekun Xülasə (Conclusions)
* Layihə uğurla reallaşdırılmış və maşın öyrənməsi ilə kredit qərarlarının verilməsi prosesinin **99.8% dəqiqliklə** avtomatlaşdırıla biləcəyi sübut edilmişdir.
* Maliyyə sektorunun biznes məntiqi ilə modelin riyazi nəticələri (CIBIL Score-un ən vacib faktor çıxması) tam olaraq üst-üstə düşür.
* Qurulmuş data və preprocessing pipeline-ı gələcəkdə yeni müştəri məlumatlarının dərhal proqnozlaşdırılması və API-lar vasitəsilə bank sisteminə qoşulması üçün tam uyğundur.
