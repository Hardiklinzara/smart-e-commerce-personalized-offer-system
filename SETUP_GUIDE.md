# SmartOffer – Setup & Deployment Guide

## What's in this folder

```
smart_ecommerce/
├── app.py                        ← Main Streamlit app
├── personalized_offer_model.pkl  ← Your trained ML model
├── requirements.txt              ← Python dependencies
├── .streamlit/
│   └── secrets.toml              ← Your MongoDB URI goes here
└── SETUP_GUIDE.md                ← This file
```

---

## STEP 1 — Set up MongoDB Atlas (free, 5 minutes)

1. Go to https://www.mongodb.com/cloud/atlas/register
2. Create a free account
3. Click **"Build a Database"** → choose **Free (M0 Sandbox)**
4. Pick any region (e.g. Mumbai) → click **Create**
5. Set a username and password → click **Create User**
6. Under "Where would you like to connect from?" → click **"Add My Current IP"** → click **Finish**
7. Click **"Connect"** → **"Connect your application"**
8. Copy the connection string — it looks like:
   ```
   mongodb+srv://youruser:yourpassword@cluster0.xxxxx.mongodb.net/
   ```
9. Replace `<password>` with your actual password in the string

---

## STEP 2 — Add MongoDB URI to Streamlit

### For local testing:
Open `.streamlit/secrets.toml` and replace the placeholder:
```toml
MONGO_URI = "mongodb+srv://youruser:yourpassword@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority"
```

### For Streamlit Cloud deployment:
- In your app dashboard → **Settings** → **Secrets**
- Paste exactly:
  ```
  MONGO_URI = "mongodb+srv://youruser:yourpassword@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority"
  ```

---

## STEP 3 — Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## STEP 4 — Deploy on Streamlit Cloud (free)

1. Push this entire folder to a GitHub repository
2. Go to https://share.streamlit.io
3. Click **"New app"** → connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **"Advanced settings"** → paste your MONGO_URI in Secrets
6. Click **Deploy** → your live URL will be ready in ~2 minutes

---

## How the Discount Engine Works

The owner sets **one number** — their maximum discount %.
The AI automatically applies the right rule per customer:

| Customer Type       | Trigger                    | Discount                    | Popup? |
|---------------------|----------------------------|-----------------------------|--------|
| 🎉 New Customer     | First-ever visit           | 50% of owner's max          | ✅ Yes |
| ❤️ Loyal Customer   | 5+ visits, regular shopper | Up to 30% of MRP            | ❌ No  |
| 🤗 Returning        | Came back after 4+ days    | Owner's max + 10% extra     | ✅ Yes |
| 🛒 Bulk Deal        | 7+ items in cart           | Full owner's max            | ❌ No  |

**No manual action needed by owner** — it's 100% automatic.

---

## MongoDB Collections Created Automatically

| Collection  | Stores                                      |
|-------------|---------------------------------------------|
| `owners`    | Shop name, password, discount %, settings   |
| `products`  | Catalog items (name, MRP, image, owner)     |
| `customers` | Visit count, last visit date, display name  |
| `purchases` | Order history per customer with discount type |

---

## Training Data / ML Model

Your `personalized_offer_model.pkl` (RandomForestClassifier) uses these features:
`age, gender, location, category, preferred_brand, color_preference, size, season,
festival, purchase_amount, purchase_frequency, avg_order_value, last_purchase_days,
browsing_time, product_views, wishlist_items, discount_used, coupon_usage_rate,
price_sensitivity, loyalty_score`

As customers shop on the live site, their data is stored in MongoDB.
You can later use this data to retrain the model with real behavioural data.
