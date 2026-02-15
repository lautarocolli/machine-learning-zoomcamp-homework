import pickle


subject = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

with open("pipeline_v1.bin", "rb") as f_out:
    pipeline = pickle.load(f_out)
pred = pipeline.predict_proba(subject)[:, 1][0]
print(f"Probability of enrollment: {pred:.3f}")