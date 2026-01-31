import joblib
import pandas as pd

# Load the trained model
model = joblib.load("ransomware_model.pkl")

# 1. Use the EXACT column names from your CSV
columns = ["cpu_usage", "file_access_rate", "process_count"]

# 2. Test with "Normal" data first to verify it works
# Try [[10, 5, 50]] for Normal or [[92, 210, 125]] for Ransomware
test_input = [[92, 210, 125]] 

# 3. Convert to DataFrame to include feature names
new_data = pd.DataFrame(test_input, columns=columns)

prediction = model.predict(new_data)

if prediction[0] == 1:
    print("⚠ Ransomware Detected!")
else:
    print("✅ System is Normal")