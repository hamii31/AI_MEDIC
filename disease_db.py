disease_db = {
    "Common Cold": {"cough", "sneezing", "sore throat", "runny nose", "congestion"},
    "Flu": {"fever", "headache", "chills", "fatigue", "body aches", "dry cough"},
    "COVID-19": {"fever", "dry cough", "loss of smell", "shortness of breath", "fatigue"},
    "Pneumonia": {"chest pain", "shortness of breath", "fever", "productive cough", "fatigue"},
    "Heart Attack": {"chest pain", "shortness of breath", "nausea", "fatigue", "cold sweat"},
    "irregular painful periods": ["PCOS"],
    "fever": ["flu", "common cold", "infection"],
    "cough": ["flu", "common cold", "bronchitis"],
    "sore throat": ["common cold", "strep throat", "sore throat"],
    "headache": ["tension headache", "migraine", "flu"],
    "shortness of breath": ["asthma", "pneumonia", "anxiety"],
    "chest pain": ["heart attack (seek immediate medical help!)", "anxiety", "muscle strain"],
    "increased thirst": ["diabetes"],
    "frequent urination": ["diabetes"],
    "unexplained weight loss": ["diabetes", "hyperthyroidism"],
    "blurred vision": ["diabetes", "hyperthyroidism"],
    "weight gain": ["hypothyroidism", "PCOS"],
    "weight loss": ["hyperthyroidism"],
    "hair loss": ["hypothyroidism", "PCOS"],
    "dry skin": ["hypothyroidism"],
    "feeling cold": ["hypothyroidism"],
    "feeling hot": ["hyperthyroidism"],
    "palpitations": ["hyperthyroidism"],
    "irregular periods": ["PCOS", "thyroid disorders"],
    "acne": ["PCOS", "hormonal imbalance"],
    "excess hair growth": ["PCOS"],
    "muscle weakness": ["hypothyroidism"],
    "constipation": ["hypothyroidism"],
    "depression": ["hypothyroidism", "diabetes"],
    "irritability": ["hyperthyroidism", "diabetes"],
    "mood swings": ["hyperthyroidism", "PCOS"],
    "cold intolerance": ["hypothyroidism"],
    "hot flashes": ["menopause", "hyperthyroidism"],
    "menstrual irregularities": ["PCOS", "thyroid disorders"],
    "fertility issues": ["PCOS", "thyroid disorders"]
}


# Original lab_tests dictionary
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
          "gender": "both"
      },
      "fasting blood glucose": {
          "unit": "mg/dL",
          "range": (70, 99),
          "condition": {
              "below": "Possible hypoglycemia",
              "above": "≥126 mg/dL indicates diabetes",
              "normal": "Normal fasting blood glucose"
          },
          "gender": "both"
      },
      "hba1c": {
          "unit": "%",
          "range": (4.0, 5.6),
          "condition": {
              "below": "Lower than normal HbA1c",
              "above": "≥6.5% indicates diabetes",
              "normal": "Normal HbA1c range"
          },
          "gender": "both"
      },
      "glucose tolerance test": {
          "unit": "mg/dL",
          "range": "less than 140 mg/dL (normal), 140-199 mg/dL (pre-diabetes), ≥200 mg/dL (diabetes)",
          "condition": {
              "below": "Below normal glucose tolerance (uncommon)",
              "above": "140-199 mg/dL indicates pre-diabetes, ≥200 mg/dL indicates diabetes",
              "normal": "Normal glucose metabolism"
          },
          "gender": "both"
      },
      "tt3": {
          "unit": "ng/dL",
          "range": (80, 180),
          "condition": {
              "below": "Possible hypothyroidism",
              "above": "Possible hyperthyroidism",
              "normal": "Normal TT3 level"
          },
          "gender": "both"
      },
      "amh": {
          "unit": "ng/mL",
          "range": (1.0, 4.0),
          "interpretation": "Normal ovarian reserve; lower in diminished reserve",
          "gender": "female"
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
          "gender": "female"
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
          "gender": "female"
      },
    "tsh": {
        "unit": "mIU/L",
        "range": (0.4, 4.0),
        "condition": {
            "below": "Possible hyperthyroidism",
            "above": "Possible hypothyroidism",
            "normal": "Within normal thyroid function range"
        },
        "gender": "both"
    },
    "ft4": {
        "unit": "pmol/L",
        "range": (12, 23),
        "condition": {
            "below": "Hypothyroidism or Pituitary disorders",
            "above": "Hyperthyroidism and Thyroiditis",
            "normal": "Normal"
        },
        "gender": "both"
    },
    "ft3": {
        "unit": "pmol/L",
        "range": (3.1, 6.9),
        "condition": {
            "above": "Hyperthyroidism and Thyroiditis",
            "below": "Hypothyroidism",
            "normal": "Normal"
        },
        "gender": "both"
    },
    "dhea-s": {
    "unit": "μg/dL",
    "range": (65, 380),
    "condition": {
        "below": "Low DHEA-S; possible adrenal insufficiency",
        "above": "Elevated DHEA-S; may indicate adrenal hyperplasia or PCOS",
        "normal": "Normal DHEA-S levels"
    },
    "gender": "both"
    },
    "testosterone": {
        "unit": "ng/dL",
        "range": (15, 70),
        "condition": {
            "below": "Low testosterone levels",
            "above": "Elevated testosterone; may indicate PCOS",
            "normal": "Normal testosterone levels"
        },
        "gender": "both"
    },
    "anti-thyroid antibodies": {
        "unit": "IU/mL",
        "range": "negative",
        "condition": {
            "normal": "Negative indicates absence of autoimmune thyroid disease",
            "above": "Positive indicates autoimmune thyroid disease"
        },
        "gender": "both"
    },
    "tat": {
        "unit": "IU/mL",
        "range": "negative",
        "condition": {
            "normal": "Negative indicates absence of thyroid autoimmune activity",
            "above": "Positive indicates thyroid autoimmune activity"
        },
        "gender": "both"
    },
    "mat": {
        "unit": "IU/mL",
        "range": "negative",
        "condition": {
            "normal": "Negative indicates absence of autoimmune thyroid disease",
            "above": "Positive indicates autoimmune thyroid disease"
        },
        "gender": "both"
    },
    "cortisol": {
        "unit": "μg/dL",
        "range": (6, 23),
        "condition": {
            "below": "Low cortisol; suggests adrenal insufficiency",
            "above": "High cortisol; suggests Cushing's syndrome",
            "normal": "Normal morning cortisol levels"
        },
        "gender": "both"
    }
}
