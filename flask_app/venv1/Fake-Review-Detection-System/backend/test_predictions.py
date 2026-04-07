"""Quick test script to verify predictions work correctly."""
import sys
from main import do_classify, validate_review

# Test cases: (review_text, expected_category)
test_cases = [
    # Genuine reviews
    ("The food here is absolutely amazing! Fresh ingredients, friendly staff, and perfect ambiance. Highly recommend!", "Genuine"),
    ("Great service and delicious pasta. Will definitely come back again soon.", "Genuine"),
    ("Best coffee in town! Barista is always friendly and remembers my order.", "Genuine"),
    
    # Suspicious/Fake reviews
    ("The best place ever!!! Amazing amazing amazing!!! Perfect 10/10!!!! Must visit!!!!", "Fake"),
    ("Worst restaurant ever. Food was terrible. Service was terrible. Everything was terrible. Never going back.", "Fake"),
    ("This place is great great great great great. Simply the best!", "Fake"),
    
    # Borderline cases
    ("The food was okay. Service was slow but friendly.", "Genuine"),
    ("Good restaurant, nice food!", "Genuine"),
]

print("=" * 80)
print("FAKE REVIEW DETECTION SYSTEM - TEST RESULTS")
print("=" * 80)

correct = 0
total = 0

for review_text, expected in test_cases:
    print(f"\nReview: {review_text[:60]}...")
    print(f"Expected: {expected}")
    
    # Validate
    is_valid, msg = validate_review(review_text)
    if not is_valid:
        print(f"❌ Validation failed: {msg}")
        continue
    
    # Classify
    meta = {"review_len": len(review_text)}
    result = do_classify(review_text, meta)
    
    prediction = result["prediction"]
    confidence = result["confidence"]
    fake_prob = result["fake_probability"]
    risk = result["risk_level"]
    
    match = "✅" if prediction == expected else "❌"
    print(f"{match} Prediction: {prediction} (Confidence: {confidence:.2%}, Risk: {risk})")
    print(f"   Fake Probability: {fake_prob:.4f}")
    
    if prediction == expected:
        correct += 1
    total += 1

print("\n" + "=" * 80)
print(f"ACCURACY: {correct}/{total} ({100*correct/total:.1f}%)")
print("=" * 80)
