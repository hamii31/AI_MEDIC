disease_db = {
    "acne": ["PCOS", "hormonal imbalance"],
    "anxiety": ["PCOS"],
    "blurred vision": ["diabetes", "hyperthyroidism"],
    "chest pain": ["heart attack (seek immediate medical help!)", "anxiety", "muscle strain"],
    "cold intolerance": ["hypothyroidism"],
    "congestion": ["common cold", "allergies"],
    "constipation": ["hypothyroidism"],
    "cough": ["flu", "common cold", "bronchitis"],
    "depression": ["hypothyroidism", "diabetes", "PCOS"],
    "dry skin": ["hypothyroidism"],
    "excess hair growth": ["PCOS"],
    "feeling cold": ["hypothyroidism"],
    "feeling hot": ["hyperthyroidism"],
    "fertility issues": ["PCOS", "thyroid disorders"],
    "fever": ["flu", "common cold", "infection"],
    "frequent urination": ["diabetes"],
    "hair loss": ["hypothyroidism", "PCOS"],
    "hypersomnia": ["PCOS"],
    "headache": ["tension headache", "migraine", "flu"],
    "hot flashes": ["menopause", "hyperthyroidism"],
    "increased thirst": ["diabetes"],
    "irregular painful periods": ["PCOS"],
    "irregular periods": ["PCOS", "thyroid disorders"],
    "irritability": ["hyperthyroidism", "diabetes"],
    "insomnia": ["PCOS"],
    "menstrual irregularities": ["PCOS", "thyroid disorders"],
    "mood swings": ["hyperthyroidism", "PCOS"],
    "muscle weakness": ["hypothyroidism"],
    "palpitations": ["hyperthyroidism"],
    "runny nose": ["common cold", "allergies"],
    "shortness of breath": ["asthma", "pneumonia", "anxiety"],
    "sneezing": ["common cold", "allergies"],
    "snoring":["PCOS"],
    "sleeping disorder": ["PCOS"],
    "sore throat": ["common cold", "strep throat", "sore throat"],
    "unexplained weight loss": ["diabetes", "hyperthyroidism"],
    "weight gain": ["hypothyroidism", "PCOS"],
    "weight loss": ["hyperthyroidism"]
}

recommendations = {
    "PCOS": (
        "For effective PCOS management, emphasizing protein-rich foods like legumes (which also provide fiber) alongside lean meats helps improve hormone balance, insulin sensitivity, and weight control. "
        "Combining this dietary approach with regular physical activity significantly enhances metabolic health, reduces symptoms, and supports overall well-being. "
        "The 2018 PCOS guideline recommends 250 minutes per week of moderate or 150 minutes per week of vigorous exercise for weight loss and weight regain prevention. "
        "Minimizing sedentary time and including strength training at least two days per week is also advised. "
        "Removing alcohol and smoking from your lifestyle is also recommended. "
        "Aim for at least 8 hours of sleep. "
        "The most beneficial interventions for PCOS include B-group vitamins (B1, B6, B12), folate (B9), inositols (B8), vitamins D, E, and K, soy isoflavones, carnitine, alpha-lipoic acid, calcium, zinc, selenium, magnesium, chromium picolinate, omega-3 fatty acids, N-acetyl-cysteine, coenzyme Q10, probiotics, quercetin, resveratrol, melatonin, cinnamon, curcumin, sage, fennel, licorice, spearmint, Chinese herbal medicine, acupuncture, and yoga. These supplements and therapies have shown improvements in insulin resistance, hormonal regulation, lipid metabolism, menstrual regularity, pregnancy rates, oxidative stress, and weight management in PCOS patients."
    )
}

lab_tests = {
      "lipid profile": {
          "unit": "mg/dL",
          "range": {
              "total cholesterol": (125, 200),
              "LDL": ("less than 100"),
              "HDL": ("greater than 40")
          },
          "condition": {
              "below": "Lipid levels lower than recommended",
              "above": "Elevated lipid levels; risk of cardiovascular disease",
              "normal": "Cardiovascular risk assessment within normal limits"
          },
      },
      "fasting blood glucose": {
          "unit": "mg/dL",
          "range": (70, 99),
          "condition": {
              "below": "Possible hypoglycemia",
              "above": "≥126 mg/dL indicates diabetes",
              "normal": "Normal fasting blood glucose"
          },
      },
      "hba1c": {
          "unit": "%",
          "range": (4.0, 5.6),
          "condition": {
              "below": "Lower than normal HbA1c",
              "above": "≥6.5% indicates diabetes",
              "normal": "Normal HbA1c range"
          },
      },
      "glucose tolerance test": {
          "unit": "mg/dL",
          "range": "less than 140 mg/dL (normal), 140-199 mg/dL (pre-diabetes), ≥200 mg/dL (diabetes)",
          "condition": {
              "below": "Below normal glucose tolerance (uncommon)",
              "above": "140-199 mg/dL indicates pre-diabetes, ≥200 mg/dL indicates diabetes",
              "normal": "Normal glucose metabolism"
          },
      },
      "tt3": {
          "unit": "ng/dL",
          "range": (80, 180),
          "condition": {
              "below": "Possible hypothyroidism",
              "above": "Possible hyperthyroidism",
              "normal": "Normal TT3 level"
          },
      },
      "amh": {
          "unit": "ng/mL",
          "range": (1.0, 4.0),
          "interpretation": "Normal ovarian reserve; lower in diminished reserve",
      },
      "lh": {
          "units": {
              "pmol/L": {
                  "interpretation": "Normal; <1.5 (Hypopituitarism, Secondary ovarian insufficiency), >12.5 (PCOS, POI, Menopause)",
                  "range": (1.5, 12.0)
              },
              "IU/L": {
                  "interpretation": "Normal;  <1.5 (Hypopituitarism, Secondary ovarian insufficiency), >9.5 (PCOS, POI, Menopause)",
                  "range": (1.5, 9.3)
              },
              "ng/mL": {
                  "interpretation": "Normal;  <0.1 (Hypopituitarism, Secondary ovarian insufficiency), >1.0 (PCOS, POI, Menopause)",
                  "range": (0.1, 1.0)
              },
          },
          "condition": {
              "below": "Hypopituitarism, Secondary ovarian insufficiency",
              "above": "PCOS, POI, Menopause",
              "normal": "Normal LH levels"
          },
      },
      "fsh": {
          "units": {
              "pmol/L": {
                  "interpretation": "Normal; <3.5 (Hypopituitarism, Secondary ovarian insufficiency), >12.5 (POI, Menopause, Turner Syndrome)",
                  "range": (3.5, 12.5)
              },
              "IU/L": {
                  "interpretation": "Normal; <1.5 (Hypopituitarism, Secondary ovarian insufficiency), >12.5 (POI, Menopause, Turner Syndrome)",
                  "range": (1.5, 12.5)
              },
              "ng/mL": {
                  "interpretation": "Normal; <0.35 (Hypopituitarism, Secondary ovarian insufficiency), >1.25 (POI, Menopause, Turner Syndrome)",
                  "range": (0.35, 1.25)
              },
          },
          "condition": {
              "below": "Hypopituitarism, Secondary ovarian insufficiency",
              "above": "POI, Menopause, Turner Syndrome",
              "normal": "Normal FSH levels"
          },
      },
    "tsh": {
        "unit": "mIU/L",
        "range": (0.4, 4.0),
        "condition": {
            "below": "Possible hyperthyroidism",
            "above": "Possible hypothyroidism",
            "normal": "Within normal thyroid function range"
        },
    },
    "ft4": {
        "unit": "pmol/L",
        "range": (12, 23),
        "condition": {
            "below": "Hypothyroidism or Pituitary disorders",
            "above": "Hyperthyroidism and Thyroiditis",
            "normal": "Normal"
        },
    },
    "ft3": {
        "unit": "pmol/L",
        "range": (3.1, 6.9),
        "condition": {
            "above": "Hyperthyroidism and Thyroiditis",
            "below": "Hypothyroidism",
            "normal": "Normal"
        },
    },
    "dhea-s": {
    "unit": "μg/dL",
    "range": (65, 380),
    "condition": {
        "below": "Low DHEA-S; possible adrenal insufficiency",
        "above": "Elevated DHEA-S; may indicate adrenal hyperplasia or PCOS",
        "normal": "Normal DHEA-S levels"
    },
    },
    "testosterone": {
        "unit": "ng/dL",
        "range": (15, 70),
        "condition": {
            "below": "Low testosterone levels",
            "above": "Elevated testosterone; may indicate PCOS",
            "normal": "Normal testosterone levels"
        },
    },
    "anti-thyroid antibodies": {
        "unit": "IU/mL",
        "range": "negative",
        "condition": {
            "normal": "Negative indicates absence of autoimmune thyroid disease",
            "above": "Positive indicates autoimmune thyroid disease"
        },
    },
    "tat": {
        "unit": "IU/mL",
        "range": "negative",
        "condition": {
            "normal": "Negative indicates absence of thyroid autoimmune activity",
            "above": "Positive indicates thyroid autoimmune activity"
        },
    },
    "mat": {
        "unit": "IU/mL",
        "range": "negative",
        "condition": {
            "normal": "Negative indicates absence of autoimmune thyroid disease",
            "above": "Positive indicates autoimmune thyroid disease"
        },
    },
    "cortisol": {
        "unit": "μg/dL",
        "range": (6, 23),
        "condition": {
            "below": "Low cortisol; suggests adrenal insufficiency",
            "above": "High cortisol; suggests Cushing's syndrome",
            "normal": "Normal morning cortisol levels"
        },
    }
}
