# FRONTEND_IMPLEMANTATION.md

## Goal
Build a **Streamlit frontend** for the already deployed Fraud Detection API without modifying the backend code or changing any existing endpoints.

Backend base URL:
`https://fraud-detection-api-wb1m.onrender.com`

API docs:
`https://fraud-detection-api-wb1m.onrender.com/docs`

The frontend must:
- collect user inputs: `ip`, `app`, `device`, `os`, `channel`, `click_time`
- send the data to the deployed backend
- display prediction results clearly as **Human / Bot**
- handle slow API response, timeout, and invalid inputs gracefully

---

## 1) Create the frontend folder

Create this structure inside the project root:

```text
frontend/
├── app.py
├── utils.py
├── requirements.txt
└── README.md
```

This frontend is only a consumer of the deployed API.

---

## 2) Add dependencies

Create `frontend/requirements.txt`:

```txt
streamlit
requests
pandas
```

Optional later additions if needed:
- `python-dateutil`
- `pydantic`

For now, keep dependencies minimal.

---

## 3) API utility layer

Create `frontend/utils.py`.

Purpose:
- keep API communication separate from the UI
- make the app cleaner and easier to debug
- centralize timeout/error handling

### Required behavior
- Base URL:
  `BASE_URL = "https://fraud-detection-api-wb1m.onrender.com"`
- POST to:
  `f"{BASE_URL}/predict"`
- send payload as JSON
- timeout should be long enough for Render cold starts, but not too long
- return parsed JSON response
- catch request failures and invalid responses

### Suggested implementation
```python
import requests

BASE_URL = "https://fraud-detection-api-wb1m.onrender.com"
TIMEOUT_SECONDS = 15


def predict(data: dict):
    url = f"{BASE_URL}/predict"
    try:
        response = requests.post(url, json=data, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()

        try:
            return response.json()
        except ValueError:
            return {
                "success": False,
                "error": "Invalid JSON response from API"
            }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Request timed out. The cloud model may be waking up."
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"API request failed: {str(e)}"
        }
```

### Notes
If the backend returns a different JSON schema, keep this utility flexible and map the returned keys inside `app.py`.

---

## 4) Streamlit UI

Create `frontend/app.py`.

### UI requirements
- Title:
  `🚨 Ad Click Fraud Detection System`
- Subtitle:
  `Detect whether a click is Human or Bot using ML`

### Sidebar inputs
Collect the following values in the sidebar:
- `ip` → `number_input`
- `app` → `number_input`
- `device` → `number_input`
- `os` → `number_input`
- `channel` → `number_input`
- `click_time` → datetime input using Streamlit-compatible approach

### Helpful UX text
Show short hints such as:
- `IP: Unique identifier of user`
- `Channel: Ad channel ID`
- `Click Time: Time of click`

### Predict button
Button label:
`🔍 Predict Fraud`

### On click flow
1. Validate inputs
2. Format datetime as string:
   `YYYY-MM-DD HH:MM:SS`
3. Build payload:

```python
{
    "ip": value,
    "app": value,
    "device": value,
    "os": value,
    "channel": value,
    "click_time": formatted_time
}
```

4. Call `utils.predict(payload)`
5. Show results

---

## 5) Streamlit datetime input handling

Streamlit does not have a native `datetime_input` in the same way that some frameworks do, so use one of these options:

### Preferred option
Use `st.date_input` and `st.time_input`, then combine them.

Example:
```python
click_date = st.date_input("Click Date")
click_time_only = st.time_input("Click Time")
```

Then combine:
```python
from datetime import datetime

click_dt = datetime.combine(click_date, click_time_only)
formatted_click_time = click_dt.strftime("%Y-%m-%d %H:%M:%S")
```

### Alternative option
Use a text input for manual datetime entry and validate the format.

Preferred option is better for usability.

---

## 6) Main app structure

A clean `app.py` should include:

- imports
- page config
- sidebar input collection
- payload preparation
- predict button handling
- success/error display
- model explanation section
- optional about section

### Suggested structure
```python
import streamlit as st
from datetime import datetime
from utils import predict

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🚨",
    layout="centered"
)

st.title("🚨 Ad Click Fraud Detection System")
st.subheader("Detect whether a click is Human or Bot using ML")

with st.sidebar:
    st.header("User Input")
    st.caption("IP: Unique identifier of user")
    ip = st.number_input("IP", min_value=0, value=87540, step=1)

    app = st.number_input("App", min_value=0, value=1, step=1)
    device = st.number_input("Device", min_value=0, value=1, step=1)
    os_value = st.number_input("OS", min_value=0, value=19, step=1)
    channel = st.number_input("Channel", min_value=0, value=130, step=1)

    st.caption("Click Time: Time of click")
    click_date = st.date_input("Click Date")
    click_time_only = st.time_input("Click Time")

    st.markdown("---")
    st.markdown("### About Project")
    st.write(
        "This app sends click details to a deployed ML API and shows whether the click looks legitimate or fraudulent."
    )

st.markdown("### Prediction Panel")

if st.button("🔍 Predict Fraud"):
    try:
        click_dt = datetime.combine(click_date, click_time_only)
        formatted_click_time = click_dt.strftime("%Y-%m-%d %H:%M:%S")

        payload = {
            "ip": int(ip),
            "app": int(app),
            "device": int(device),
            "os": int(os_value),
            "channel": int(channel),
            "click_time": formatted_click_time,
        }

        with st.spinner("⏳ Sending request to cloud model..."):
            result = predict(payload)

        if not result.get("success", True) and "error" in result:
            st.error(result["error"])
        else:
            st.success("Prediction received successfully")

            # Map API response keys here depending on backend output
            prediction = result.get("prediction")
            probability = result.get("probability")
            model_name = result.get("model", "XGBoost")

            if probability is not None:
                st.progress(float(probability))
                st.write(f"Fraud Probability: {round(float(probability) * 100, 2)}%")

            if prediction in [1, "1", True, "fraud", "Fraud", "bot", "Bot"]:
                st.error("🔴 Fraudulent Click")
            else:
                st.success("🟢 Legitimate Click")

            st.info(f"Model used: {model_name}")

    except Exception as e:
        st.error(f"Service temporarily unavailable: {str(e)}")
```

---

## 7) Prediction mapping

Because backend response formats can vary, keep the frontend tolerant.

Possible response fields might include:
- `prediction`
- `predicted_class`
- `label`
- `probability`
- `fraud_probability`
- `model`
- `success`
- `message`

Add a small mapping layer in `app.py` if needed:

```python
prediction = (
    result.get("prediction")
    or result.get("predicted_class")
    or result.get("label")
)

probability = (
    result.get("probability")
    or result.get("fraud_probability")
)
```

Then normalize values before displaying.

---

## 8) Output display rules

### If response is successful
Show:
- probability as a percentage
- a progress bar
- colored result message
- model name, ideally `XGBoost`

### If response fails
Show:
- `Service temporarily unavailable`
- a helpful message about cold start or API downtime

### If input is invalid
Prevent API call and show validation message using `st.warning` or `st.error`.

---

## 9) Input validation suggestions

Before sending the request:
- ensure all numeric fields are non-negative
- ensure datetime is present
- ensure field values are integers where expected

Example validation:
```python
if ip < 0 or app < 0 or device < 0 or os_value < 0 or channel < 0:
    st.error("Please enter valid non-negative values.")
    st.stop()
```

---

## 10) Optional enhancements

Add these only if they do not make the app heavier:

- show prediction in a styled card-like section
- display a compact input summary
- add a small example payload
- add `st.metric` for probability
- add a note that the model is served remotely on Render
- use `st.toast` for success feedback

Example summary box:
```python
st.info(
    f"Input summary: IP={ip}, App={app}, Device={device}, OS={os_value}, Channel={channel}, Click Time={formatted_click_time}"
)
```

---

## 11) README file

Create `frontend/README.md` with:
- project description
- setup steps
- run command
- notes about API endpoint
- troubleshooting tips

Example content:

```md
# Fraud Detection Frontend

This is a Streamlit frontend for the deployed Fraud Detection API.

## Setup
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

## API Used
`https://fraud-detection-api-wb1m.onrender.com/predict`

## Notes
- No backend changes are required.
- The app only consumes the deployed API.
- If the API is cold-starting, the first request may take a few seconds.
```

---

## 12) Run locally

From the project root:

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

Then Streamlit opens in the browser.

---

## 13) Expected user flow

1. Browser opens the Streamlit app
2. User enters click details
3. User clicks `Predict Fraud`
4. Frontend sends request to Render API
5. API returns prediction
6. App shows whether the click is Human or Bot

---

## 14) Completion checklist

- [ ] `frontend/` folder created
- [ ] `requirements.txt` added
- [ ] `utils.py` added with API call logic
- [ ] `app.py` added with Streamlit UI
- [ ] `README.md` added
- [ ] API request works without backend changes
- [ ] prediction output displays properly
- [ ] timeout and failure cases handled

---

## 15) Important constraint

Do **not** change any backend code, model code, or API route definitions.
This frontend must remain a pure consumer of the deployed service.

---

## 16) Viva-ready explanation

**Question:** What does the frontend do?

**Answer:**
The Streamlit frontend collects user input and sends it to the deployed FastAPI backend, which performs real-time fraud prediction using the trained model.

---

## 17) Final architecture

```text
User → Streamlit Frontend → Render FastAPI Backend → ML Model Prediction → Result Displayed in Browser
```

This keeps the system modular, clean, and safe for deployment.

