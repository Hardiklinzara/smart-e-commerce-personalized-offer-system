import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import joblib, os, warnings

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartOffer – AI Personalised Commerce",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── MONGODB ──────────────────────────────────────────────────────────────────
from pymongo import MongoClient
from bson import ObjectId

@st.cache_resource
def get_db():
    uri = ""
    if hasattr(st, "secrets") and "MONGO_URI" in st.secrets:
        uri = st.secrets["MONGO_URI"]
    elif os.environ.get("MONGO_URI"):
        uri = os.environ["MONGO_URI"]
    if not uri:
        st.error("❌  MongoDB URI not configured. Add MONGO_URI to Streamlit Secrets.")
        st.stop()
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    return client["smart_ecommerce"]

def col(name):
    return get_db()[name]

# ─── ML MODEL ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), "personalized_offer_model.pkl")
    if os.path.exists(path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return joblib.load(path)
    return None

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html,body,[class*="css"]{ font-family:'Inter',sans-serif; }
#MainMenu,footer,header{ visibility:hidden; }
.block-container{ padding-top:1.2rem; padding-bottom:2rem; }

:root{
  --bg:#f8f9fa; --surf:#ffffff; --bdr:#e9ecef;
  --acc:#2563eb; --grn:#16a34a; --red:#dc2626; --ylw:#d97706; --pur:#7c3aed;
  --txt:#111827; --mut:#6b7280;
}
/* dark mode overrides */
@media (prefers-color-scheme: dark) {
  :root{
    --surf:#1e1e2e; --bdr:#2e2e42;
    --txt:#f1f1f1; --mut:#a0a0b0;
  }
}

.topbar{
  display:flex; align-items:center; justify-content:space-between;
  padding:.7rem 1.3rem; background:var(--surf);
  border:1px solid var(--bdr); border-radius:12px; margin-bottom:1.2rem;
}
.topbar-brand{ font-size:1.05rem; font-weight:700; color:var(--txt); }
.topbar-right{ font-size:.84rem; color:var(--mut); display:flex; align-items:center; gap:.6rem; }

.bdg{
  display:inline-block; padding:.18rem .55rem; border-radius:50px;
  font-size:.72rem; font-weight:600;
}
.bdg-blue  { background:#eff6ff; color:var(--acc); }
.bdg-green { background:#f0fdf4; color:var(--grn); }
.bdg-ylw   { background:#fefce8; color:var(--ylw); }
.bdg-pur   { background:#fdf4ff; color:var(--pur); }

.p-card{
  background:var(--surf); border:1px solid var(--bdr);
  border-radius:14px; overflow:hidden;
  transition:box-shadow .2s,transform .2s; margin-bottom:1rem;
}
.p-card:hover{ box-shadow:0 8px 24px rgba(0,0,0,.08); transform:translateY(-2px); }
.p-body{ padding:.75rem .9rem .9rem; }
.p-name{ font-size:.95rem; font-weight:600; color:var(--txt); margin-bottom:.15rem; }
.p-shop{ font-size:.75rem; color:var(--mut); margin-bottom:.45rem; }
.p-prices{ display:flex; align-items:baseline; gap:.45rem; flex-wrap:wrap; }
.p-mrp  { text-decoration:line-through; color:var(--mut); font-size:.85rem; }
.p-final{ font-size:1.1rem; font-weight:700; color:var(--txt); }
.p-save { font-size:.75rem; color:var(--grn); font-weight:500; }

.lcard{
  max-width:430px; margin:3rem auto;
  background:#ffffff !important; border:1px solid #e9ecef;
  border-radius:18px; padding:2.5rem 2rem 2rem;
  box-shadow:0 4px 24px rgba(0,0,0,0.12);
}
.lcard-logo{ font-size:2.2rem; text-align:center; margin-bottom:.2rem; }
.lcard-title{
  font-size:1.7rem; font-weight:800;
  color:#2563eb !important;
  text-align:center; letter-spacing:-0.5px;
  margin-bottom:.3rem;
}
.lcard-sub{ font-size:.87rem; color:#6b7280 !important; text-align:center; margin-bottom:1.8rem; }

.cart-box{
  background:var(--surf); border:1px solid var(--bdr);
  border-radius:14px; padding:1rem 1.1rem;
}
.cart-title{ font-weight:700; font-size:.95rem; margin-bottom:.7rem; }
.cart-item{
  display:flex; justify-content:space-between;
  padding:.4rem 0; border-bottom:1px solid var(--bdr);
  font-size:.84rem; color:var(--txt);
}
.cart-item:last-child{ border-bottom:none; }
.cart-total{
  font-size:1.1rem; font-weight:700; color:var(--txt);
  margin-top:.65rem; padding-top:.65rem;
  border-top:2px solid var(--bdr);
}
.saving-pill{
  background:#f0fdf4; border:1px solid #bbf7d0; color:var(--grn);
  font-size:.8rem; font-weight:600; border-radius:50px;
  padding:.25rem .75rem; margin-top:.5rem; display:inline-block;
}

.pop-overlay{
  position:fixed; inset:0; background:rgba(0,0,0,.5);
  z-index:9999; display:flex; align-items:center; justify-content:center;
}
.pop-box{
  background:#fff; border-radius:22px; padding:2.5rem 2rem;
  max-width:400px; width:90%; text-align:center;
  animation:popIn .4s cubic-bezier(.34,1.56,.64,1);
  box-shadow:0 25px 60px rgba(0,0,0,.18);
}
@keyframes popIn{
  from{ transform:scale(.55); opacity:0; }
  to  { transform:scale(1);   opacity:1; }
}
.pop-emoj { font-size:3.8rem; line-height:1.1; }
.pop-title{ font-size:1.6rem; font-weight:800; color:var(--txt); margin:.4rem 0 .2rem; }
.pop-sub  { font-size:.95rem; color:var(--mut); }
.pop-disc { font-size:3rem; font-weight:900; color:var(--acc); margin:.3rem 0; }
.pop-note { font-size:.8rem; color:var(--mut); margin-top:.3rem; }
.confetti { font-size:1.3rem; letter-spacing:.2rem; }

.adm-card{
  background:var(--surf); border:1px solid var(--bdr);
  border-radius:14px; padding:1.3rem 1.4rem; margin-bottom:1rem;
}
.adm-card h4{ font-size:1rem; font-weight:700; color:var(--txt); margin:0 0 .9rem; }

.stats-row{ display:flex; gap:.6rem; flex-wrap:wrap; margin-bottom:1rem; }
.stat-chip{
  background:var(--surf); border:1px solid var(--bdr);
  border-radius:10px; padding:.5rem .95rem;
  font-size:.84rem; color:var(--txt);
}
.stat-chip b{ color:var(--acc); }

div.stButton>button{
  border-radius:9px; font-weight:500;
  border:1.5px solid var(--bdr);
  background:var(--surf); color:var(--txt);
  transition:all .15s; font-size:.88rem;
}
div.stButton>button:hover{
  border-color:var(--acc); color:var(--acc); background:#eff6ff;
}
div[data-testid="stBaseButton-primary"]>button{
  background:var(--acc) !important; color:#fff !important;
  border-color:var(--acc) !important;
}
div[data-testid="stBaseButton-primary"]>button:hover{
  background:#1d4ed8 !important;
}
div[data-baseweb="input"] input,
div[data-baseweb="select"]>div,
div[data-baseweb="textarea"] textarea{
  border-radius:9px !important; border-color:var(--bdr) !important;
}
hr.thin{ border:none; border-top:1px solid var(--bdr); margin:.9rem 0; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  DB HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def now():
    return datetime.utcnow()

def get_owner(name):
    return col("owners").find_one({"name": name.strip().lower()})

def save_owner(data):
    col("owners").update_one({"name": data["name"]}, {"$set": data}, upsert=True)

def get_all_owners():
    return list(col("owners").find({}))

def get_products(owner_name):
    return list(col("products").find({"owner": owner_name.lower()}))

def add_product(p):
    col("products").insert_one(p)

def delete_product(pid):
    col("products").delete_one({"_id": ObjectId(pid)})

def update_product(pid, data):
    col("products").update_one({"_id": ObjectId(pid)}, {"$set": data})

def get_customer(username):
    return col("customers").find_one({"username": username.strip().lower()})

def upsert_customer(username, data):
    col("customers").update_one(
        {"username": username.strip().lower()},
        {"$set": data}, upsert=True
    )

def log_purchase(username, owner, items, total, dtype):
    col("purchases").insert_one({
        "username": username.lower(), "owner": owner.lower(),
        "items": items, "total": total,
        "discount_type": dtype, "ts": now()
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  AUTOMATIC DISCOUNT ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def classify_customer(username):
    """
    Returns (type, label, badge_class)
    Fully automatic — based on DB behaviour data only.
    """
    profile = get_customer(username)

    if not profile or profile.get("visit_count", 0) == 0:
        return "new", "New Customer 🎉", "bdg-blue"

    last_visit  = profile.get("last_visit")
    visit_count = profile.get("visit_count", 0)

    if last_visit:
        if isinstance(last_visit, str):
            last_visit = datetime.fromisoformat(last_visit)
        days_away = (now() - last_visit).days
    else:
        days_away = 0

    if days_away >= 4:
        return "returning", "Welcome Back 🤗", "bdg-ylw"

    if visit_count >= 5:
        return "loyal", "Loyal Customer ❤️", "bdg-green"

    return "standard", "Valued Customer ⭐", "bdg-blue"


def compute_discount(owner, ctype, cart_qty=0):
    """
    Owner sets ONE number: their max discount %.
    All 4 rules apply automatically:
      Rule 1 – New       → 50% of owner's max  (welcome offer)
      Rule 2 – Loyal     → min(30, owner's max) (loyalty reward on MRP)
      Rule 3 – Returning → owner's max + 10%    (comeback nudge)
      Rule 4 – Bulk 7+   → full owner's max     (bulk deal)
    """
    base = float(owner.get("discount_pct", 20))

    if cart_qty >= 7:
        return round(base, 1)                        # Rule 4

    if ctype == "new":
        return round(base * 0.50, 1)                 # Rule 1

    if ctype == "loyal":
        return round(min(base, 30.0), 1)             # Rule 2

    if ctype == "returning":
        return round(min(base + 10.0, 80.0), 1)      # Rule 3

    return round(base * 0.40, 1)                     # standard


def is_bumper(ctype):
    return ctype in ("new", "returning")


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

_defaults = dict(
    role=None, username="", owner_name="",
    cart={}, popup_shown=False,
    ctype=None, clabel="", cbadge="",
    active_owner=None,
)
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════════════════
#  LOGIN
# ═══════════════════════════════════════════════════════════════════════════════

def login_page():
    st.markdown('<div class="lcard">', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; margin-bottom:1.5rem;">
      <div style="font-size:2.4rem; margin-bottom:.3rem;">🛍️</div>
      <div style="font-size:1.8rem; font-weight:800; color:#2563eb;
                  letter-spacing:-0.5px; margin-bottom:.3rem;">SmartOffer</div>
      <div style="font-size:.87rem; color:#6b7280;">
        AI-powered personalised offers &middot; Every shopper, every visit
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab_cust, tab_owner = st.tabs(["🛒  Customer", "🏪  Shop Owner"])

    with tab_cust:
        _customer_login()

    with tab_owner:
        _owner_login()

    st.markdown('</div>', unsafe_allow_html=True)


def _customer_login():
    owners = get_all_owners()
    username = st.text_input("Your Name", placeholder="e.g. Priya Sharma", key="c_name")

    if not owners:
        st.info("No shops registered yet. Ask a business owner to sign up.")
        return

    shop = st.selectbox("Choose a Shop", [o["display_name"] for o in owners], key="c_shop")

    if st.button("Start Shopping →", type="primary", use_container_width=True, key="c_login"):
        if not username.strip():
            st.warning("Please enter your name.")
            return

        owner = next(o for o in owners if o["display_name"] == shop)

        # classify BEFORE updating visit count (so first-ever visit = new)
        ctype, clabel, cbadge = classify_customer(username.strip())

        profile = get_customer(username.strip())
        visit_count = (profile or {}).get("visit_count", 0) + 1
        upsert_customer(username.strip(), {
            "last_visit": now().isoformat(),
            "visit_count": visit_count,
            "display_name": username.strip(),
        })

        st.session_state.update(dict(
            role="customer", username=username.strip(),
            active_owner=owner,
            ctype=ctype, clabel=clabel, cbadge=cbadge,
            cart={}, popup_shown=False,
        ))
        st.rerun()


def _owner_login():
    st.markdown("**Sign in to your shop**")
    oname = st.text_input("Shop Name", key="o_name")
    opwd  = st.text_input("Password", type="password", key="o_pwd")

    if st.button("Owner Login →", type="primary", use_container_width=True, key="o_login"):
        if not oname.strip() or not opwd.strip():
            st.warning("Fill all fields.")
            return
        owner = get_owner(oname.strip())
        if not owner or owner.get("password") != opwd:
            st.error("Invalid shop name or password.")
            return
        st.session_state.update(dict(
            role="owner", owner_name=oname.strip(), active_owner=owner,
        ))
        st.rerun()

    st.markdown('<hr class="thin">', unsafe_allow_html=True)
    st.markdown("**New here? Register your shop**")

    with st.form("reg_form"):
        rname = st.text_input("Shop / Business Name *", key="r_name")
        rpwd  = st.text_input("Create Password *", type="password", key="r_pwd")
        rdisc = st.slider(
            "Maximum discount you allow on MRP (%)",
            min_value=5, max_value=80, value=20, step=5,
            help="Set this once. The AI automatically splits it across all 4 customer rules."
        )
        reg_submitted = st.form_submit_button("Register Shop →", use_container_width=True)

    if reg_submitted:
        if not rname.strip() or not rpwd.strip():
            st.warning("Shop name and password are required.")
        elif get_owner(rname.strip()):
            st.error("Shop name already registered. Try a different name.")
        else:
            save_owner({
                "name": rname.strip().lower(),
                "display_name": rname.strip(),
                "password": rpwd,
                "discount_pct": rdisc,
                "registered_at": now().isoformat(),
            })
            st.success(f"✅ '{rname.strip()}' registered! You can now log in above.")


# ═══════════════════════════════════════════════════════════════════════════════
#  CELEBRATION POPUP
# ═══════════════════════════════════════════════════════════════════════════════

def show_popup():
    owner = st.session_state.active_owner
    ctype = st.session_state.ctype
    uname = st.session_state.username
    disc  = compute_discount(owner, ctype)

    if ctype == "new":
        emoj     = "🎉 🎊 🥳"
        title    = f"Welcome, {uname}!"
        sub      = "You've unlocked an exclusive first-time offer"
        confetti = "🎈 🎁 🎀 🎊 🎉"
    else:
        emoj     = "🎈 🎁 🎀"
        title    = f"Welcome Back, {uname}!"
        sub      = "We missed you — here's a special comeback reward"
        confetti = "🥳 🎊 🎉 🎈 🎁"

    # Render popup HTML (visual only — button is rendered below via Streamlit)
    st.markdown(f"""
    <div style="background:rgba(0,0,0,0.55);position:fixed;inset:0;z-index:9998;
                display:flex;align-items:center;justify-content:center;">
      <div style="background:#fff;border-radius:22px;padding:2.5rem 2rem 1.5rem;
                  max-width:400px;width:90%;text-align:center;
                  box-shadow:0 25px 60px rgba(0,0,0,.22);">
        <div style="font-size:3.5rem;line-height:1.1;">{emoj}</div>
        <div style="font-size:1.3rem;letter-spacing:.15rem;margin:.3rem 0;">{confetti}</div>
        <div style="font-size:1.6rem;font-weight:800;color:#111827;margin:.4rem 0 .2rem;">{title}</div>
        <div style="font-size:.95rem;color:#6b7280;">{sub}</div>
        <div style="font-size:3rem;font-weight:900;color:#2563eb;margin:.3rem 0;">{disc}% OFF</div>
        <div style="font-size:.8rem;color:#9ca3af;margin-bottom:1.2rem;">
          Applied automatically to every item in your cart 🛒
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Streamlit button rendered below the overlay — always clickable
    st.markdown("<div style='height:340px'></div>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        if st.button("🎉  Claim & Start Shopping!",
                     type="primary", use_container_width=True, key="popup_close"):
            st.session_state.popup_shown = True
            st.rerun()
    if st.button("✖  Skip offer and browse", use_container_width=True, key="popup_skip"):
        st.session_state.popup_shown = True
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  CUSTOMER DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def customer_dashboard():
    owner = st.session_state.active_owner
    ctype = st.session_state.ctype

    if not st.session_state.popup_shown and is_bumper(ctype):
        show_popup()
        return

    cart     = st.session_state.cart
    cart_qty = sum(cart.values())
    eff_type = "bulk" if cart_qty >= 7 else ctype
    disc_pct = compute_discount(owner, eff_type, cart_qty)

    badge_map = {
        "new": "bdg-blue", "loyal": "bdg-green",
        "returning": "bdg-ylw", "bulk": "bdg-pur", "standard": "bdg-blue"
    }
    label = "🛒 Bulk Deal!" if eff_type == "bulk" else st.session_state.clabel

    # topbar with inline logout
    tcol1, tcol2 = st.columns([6, 1])
    with tcol1:
        st.markdown(f"""
        <div class="topbar">
          <span class="topbar-brand">🛍️ {owner['display_name']}</span>
          <span class="topbar-right">
            Hi <b>{st.session_state.username}</b>
            <span class="bdg {badge_map.get(eff_type,'bdg-blue')}">{label}</span>
            <span style="background:#eff6ff;color:#2563eb;font-weight:700;
                         padding:.2rem .6rem;border-radius:50px;font-size:.8rem;">
              {disc_pct}% OFF active
            </span>
          </span>
        </div>
        """, unsafe_allow_html=True)
    with tcol2:
        if st.button("🚪 Logout", use_container_width=True, key="cust_logout"):
            for k, v in _defaults.items():
                st.session_state[k] = v
            st.rerun()

    prod_col, cart_col = st.columns([3, 1.15], gap="large")

    # ── Products ──────────────────────────────────────────────────────────────
    with prod_col:
        search = st.text_input(
            "Search", placeholder="🔍  Search products…",
            label_visibility="collapsed", key="srch"
        )
        products = get_products(owner["name"])

        if not products:
            st.info("This shop hasn't added any products yet. Check back soon!")
        else:
            visible = [p for p in products
                       if not search or search.lower() in p["name"].lower()]

            if not visible:
                st.info("No products match your search.")
            else:
                gcols = st.columns(3, gap="small")
                for i, p in enumerate(visible):
                    mrp   = float(p.get("mrp", 0))
                    final = round(mrp * (1 - disc_pct / 100))
                    saved = round(mrp - final)
                    pid   = str(p["_id"])
                    qty   = cart.get(pid, 0)

                    with gcols[i % 3]:
                        img = p.get("image_url", "")
                        if img:
                            st.image(img, use_container_width=True)
                        else:
                            st.markdown(
                                '<div style="background:#f1f5f9;border-radius:10px;'
                                'height:110px;display:flex;align-items:center;'
                                'justify-content:center;font-size:2rem;">📦</div>',
                                unsafe_allow_html=True
                            )

                        st.markdown(f"""
                        <div class="p-body">
                          <div class="p-name">{p['name']}</div>
                          <div class="p-shop">by {owner['display_name']}</div>
                          <div class="p-prices">
                            <span class="p-mrp">₹{int(mrp)}</span>
                            <span class="p-final">₹{int(final)}</span>
                          </div>
                          <div class="p-save">Save ₹{int(saved)}</div>
                          <span class="bdg {badge_map.get(eff_type,'bdg-blue')}"
                                style="margin-top:.35rem;">
                            {disc_pct}% OFF · {label}
                          </span>
                        </div>
                        """, unsafe_allow_html=True)

                        if qty == 0:
                            if st.button("Add to Cart", key=f"add_{pid}",
                                         use_container_width=True):
                                st.session_state.cart[pid] = 1
                                st.rerun()
                        else:
                            b1, b2, b3 = st.columns([1, 1.4, 1])
                            with b1:
                                if st.button("−", key=f"d_{pid}"):
                                    st.session_state.cart[pid] -= 1
                                    if st.session_state.cart[pid] <= 0:
                                        del st.session_state.cart[pid]
                                    st.rerun()
                            with b2:
                                st.markdown(
                                    f'<div style="text-align:center;font-weight:700;'
                                    f'padding:.32rem 0;">{qty}</div>',
                                    unsafe_allow_html=True
                                )
                            with b3:
                                if st.button("+", key=f"i_{pid}"):
                                    st.session_state.cart[pid] += 1
                                    st.rerun()

    # ── Cart ──────────────────────────────────────────────────────────────────
    with cart_col:
        _render_cart(products if products else [], disc_pct, eff_type)


def _render_cart(products, disc_pct, eff_type):
    cart  = st.session_state.cart
    owner = st.session_state.active_owner

    # ── Cart box with forced-visible colours ──────────────────────────────────
    st.markdown("""
    <div style="background:#ffffff;border:1px solid #e9ecef;border-radius:14px;
                padding:1rem 1.1rem;margin-bottom:.8rem;">
      <div style="font-weight:700;font-size:.95rem;color:#111827;margin-bottom:.7rem;">
        🧾 Your Cart
      </div>
    """, unsafe_allow_html=True)

    if not cart:
        st.markdown(
            '<p style="color:#9ca3af;font-size:.85rem;margin:0;">Cart is empty</p>'
            '</div>', unsafe_allow_html=True
        )
        return

    total     = 0
    total_mrp = 0
    cart_items = []

    for pid, qty in list(cart.items()):
        prod = next((p for p in products if str(p["_id"]) == pid), None)
        if not prod:
            continue
        mrp   = float(prod.get("mrp", 0))
        final = round(mrp * (1 - disc_pct / 100))
        line  = final * qty
        total     += line
        total_mrp += mrp * qty
        cart_items.append({"name": prod["name"], "qty": qty, "price": final})

        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:flex-start;
                    padding:.4rem 0;border-bottom:1px solid #e9ecef;">
          <span style="color:#111827;font-size:.84rem;">
            {prod['name']}<br>
            <span style="color:#9ca3af;font-size:.75rem;">×{qty} · ₹{int(final)} each</span>
          </span>
          <span style="font-weight:600;color:#111827;font-size:.84rem;">₹{int(line)}</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Billing breakdown ─────────────────────────────────────────────────────
    subtotal   = round(total_mrp)
    discount_amt = round(total_mrp - total)
    gst_amt    = round(total * 0.18)
    grand_total = round(total + gst_amt)

    st.markdown(f"""
    <div style="margin-top:.8rem;padding-top:.6rem;border-top:1px solid #e9ecef;">
      <div style="display:flex;justify-content:space-between;
                  font-size:.83rem;color:#6b7280;padding:.25rem 0;">
        <span>Subtotal (MRP)</span>
        <span>₹{int(subtotal)}</span>
      </div>
      <div style="display:flex;justify-content:space-between;
                  font-size:.83rem;color:#16a34a;padding:.25rem 0;font-weight:600;">
        <span>Discount ({disc_pct}% off)</span>
        <span>− ₹{int(discount_amt)}</span>
      </div>
      <div style="display:flex;justify-content:space-between;
                  font-size:.83rem;color:#6b7280;padding:.25rem 0;">
        <span>GST (18%)</span>
        <span>+ ₹{int(gst_amt)}</span>
      </div>
      <div style="display:flex;justify-content:space-between;
                  font-size:1.05rem;font-weight:800;color:#111827;
                  padding:.55rem 0 .2rem;border-top:2px solid #111827;margin-top:.3rem;">
        <span>Grand Total</span>
        <span>₹{int(grand_total)}</span>
      </div>
      <div style="background:#f0fdf4;border:1px solid #bbf7d0;color:#16a34a;
                  font-size:.8rem;font-weight:600;border-radius:50px;
                  padding:.25rem .75rem;margin-top:.5rem;display:inline-block;">
        🎉 You save ₹{int(discount_amt)} on this order!
      </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    if st.button("✅  Place Order", type="primary", use_container_width=True, key="checkout"):
        log_purchase(
            st.session_state.username,
            owner["name"],
            cart_items, total, eff_type
        )
        st.session_state.cart = {}
        st.success("🎉 Order placed! Thank you.")
        st.rerun()

    if st.button("🗑  Clear Cart", use_container_width=True, key="clear_cart"):
        st.session_state.cart = {}
        st.rerun()

    # bulk progress nudge
    total_qty = sum(cart.values())
    if total_qty < 7:
        remaining = 7 - total_qty
        st.markdown(f"""
        <div style="margin-top:.9rem;background:#fdf4ff;border:1px solid #e9d5ff;
                    border-radius:10px;padding:.6rem .8rem;font-size:.8rem;color:#7c3aed;">
          🛒 Add <b>{remaining}</b> more item(s) to unlock
          <b>{owner.get('discount_pct',20)}% Bulk Deal!</b>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  OWNER DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def owner_dashboard():
    # reload fresh data
    fresh = get_owner(st.session_state.active_owner["name"])
    if fresh:
        st.session_state.active_owner = fresh
    owner = st.session_state.active_owner
    disc  = float(owner.get("discount_pct", 20))

    # computed rule previews
    new_d  = round(disc * 0.50, 1)
    loy_d  = min(disc, 30.0)
    ret_d  = min(disc + 10.0, 80.0)
    blk_d  = disc

    otcol1, otcol2 = st.columns([6, 1])
    with otcol1:
        st.markdown(f"""
        <div class="topbar">
          <span class="topbar-brand">🏪 {owner['display_name']} — Owner Panel</span>
          <span class="topbar-right">Max Discount: <b style="color:#2563eb;">{disc}%</b></span>
        </div>
        """, unsafe_allow_html=True)
    with otcol2:
        if st.button("🚪 Logout", use_container_width=True, key="owner_logout"):
            for k, v in _defaults.items():
                st.session_state[k] = v
            st.rerun()

    # auto-rules card
    st.markdown(f"""
    <div class="adm-card" style="background:#f8faff;border-color:#dbeafe;">
      <h4 style="color:#1d4ed8;">⚡ Auto-Applied Discount Rules</h4>
      <div class="stats-row">
        <div class="stat-chip">🎉 New Customer <b>{new_d}%</b>
          <span style="color:#9ca3af;font-size:.73rem;">&nbsp;50% of your max</span></div>
        <div class="stat-chip">❤️ Loyal Customer <b>{loy_d}%</b>
          <span style="color:#9ca3af;font-size:.73rem;">&nbsp;30% of MRP</span></div>
        <div class="stat-chip">🤗 Returning (4+ days) <b>{ret_d}%</b>
          <span style="color:#9ca3af;font-size:.73rem;">&nbsp;your max + 10%</span></div>
        <div class="stat-chip">🛒 Bulk (7+ items) <b>{blk_d}%</b>
          <span style="color:#9ca3af;font-size:.73rem;">&nbsp;your full max</span></div>
      </div>
      <span style="font-size:.78rem;color:#6b7280;">
        All rules are fully automatic. Customers get the right discount based on their behaviour — no manual action needed.
      </span>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📦  Catalog", "➕  Add Product", "⚙️  Settings", "📊  Analytics"]
    )

    # ── CATALOG ──────────────────────────────────────────────────────────────
    with tab1:
        products = get_products(owner["name"])
        if not products:
            st.info("No products yet — add some in the 'Add Product' tab.")
        else:
            st.markdown(
                f'<div style="color:#6b7280;font-size:.85rem;margin-bottom:.8rem;">'
                f'{len(products)} product(s) in catalog</div>',
                unsafe_allow_html=True
            )
            gcols = st.columns(3, gap="small")
            for i, p in enumerate(products):
                pid = str(p["_id"])
                mrp = float(p.get("mrp", 0))
                with gcols[i % 3]:
                    img = p.get("image_url", "")
                    if img:
                        st.image(img, use_container_width=True)
                    else:
                        st.markdown(
                            '<div style="background:#f1f5f9;border-radius:10px;'
                            'height:90px;display:flex;align-items:center;'
                            'justify-content:center;font-size:1.8rem;">📦</div>',
                            unsafe_allow_html=True
                        )
                    st.markdown(f"""
                    <div style="padding:.5rem 0;">
                      <div style="font-weight:600;font-size:.9rem;">{p['name']}</div>
                      <div style="color:#6b7280;font-size:.8rem;">MRP ₹{int(mrp)}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("Edit"):
                        new_name = st.text_input("Name", value=p["name"], key=f"en_{pid}")
                        new_mrp  = st.number_input("MRP (₹)", value=mrp,
                                                    key=f"em_{pid}", min_value=1.0)
                        new_img  = st.text_input("Image URL",
                                                  value=p.get("image_url",""),
                                                  key=f"ei_{pid}")
                        if st.button("💾 Save", key=f"esv_{pid}", use_container_width=True):
                            update_product(pid, {
                                "name": new_name, "mrp": new_mrp, "image_url": new_img
                            })
                            st.success("Updated!")
                            st.rerun()

                    if st.button("🗑 Delete", key=f"del_{pid}", use_container_width=True):
                        delete_product(pid)
                        st.rerun()

    # ── ADD PRODUCT ──────────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="adm-card"><h4>➕ Add New Product</h4>',
                    unsafe_allow_html=True)
        with st.form("add_prod"):
            pname = st.text_input("Product Name *")
            pmrp  = st.number_input("MRP / Original Price (₹) *",
                                     min_value=1.0, value=999.0)
            pimg  = st.text_input("Image URL",
                                   placeholder="https://…  (optional)")
            pdesc = st.text_area("Description (optional)", height=80)
            add_submitted = st.form_submit_button(
                "Add to Catalog", use_container_width=True
            )

        if add_submitted:
            if not pname.strip():
                st.warning("Product name is required.")
            else:
                add_product({
                    "owner": owner["name"],
                    "name": pname.strip(),
                    "mrp": pmrp,
                    "image_url": pimg.strip(),
                    "description": pdesc.strip(),
                    "added_at": now().isoformat(),
                })
                st.success(f"✅ '{pname.strip()}' added!")
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── SETTINGS ─────────────────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="adm-card"><h4>⚙️ Shop Settings</h4>',
                    unsafe_allow_html=True)
        with st.form("settings"):
            new_disc_val = st.slider(
                "Maximum Discount on MRP (%)",
                min_value=5, max_value=80,
                value=int(disc), step=5,
                help=(
                    "This is the only value you need to set. "
                    "All 4 discount rules are derived automatically from this number."
                )
            )
            new_display = st.text_input("Shop Display Name",
                                         value=owner["display_name"])
            new_pwd = st.text_input(
                "New Password (leave blank to keep current)",
                type="password"
            )
            save_set = st.form_submit_button("Save Settings",
                                              use_container_width=True)

        if save_set:
            updates = {
                "discount_pct": new_disc_val,
                "display_name": new_display
            }
            if new_pwd.strip():
                updates["password"] = new_pwd.strip()
            save_owner({**owner, **updates})
            st.success("✅ Settings saved!")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── ANALYTICS ────────────────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="adm-card"><h4>📊 Customer Behaviour Analytics</h4>',
                    unsafe_allow_html=True)
        purchases = list(
            col("purchases").find({"owner": owner["name"]}).sort("ts", -1)
        )

        if not purchases:
            st.info("No purchases recorded yet.")
        else:
            df = pd.DataFrame(purchases)
            df["total"] = pd.to_numeric(df["total"], errors="coerce")
            df["date"]  = pd.to_datetime(df["ts"]).dt.date

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Orders",    len(df))
            c2.metric("Revenue",         f"₹{int(df['total'].sum()):,}")
            c3.metric("Avg Order Value", f"₹{int(df['total'].mean()):,}")
            c4.metric("Unique Shoppers", df["username"].nunique())

            st.markdown('<hr class="thin">', unsafe_allow_html=True)
            st.markdown("**Orders by Discount Type**")
            dt = df["discount_type"].value_counts().reset_index()
            dt.columns = ["Discount Rule", "Orders"]
            st.dataframe(dt, use_container_width=True, hide_index=True)

            st.markdown("**Recent Orders**")
            recent = df[["date","username","total","discount_type"]].head(25).copy()
            recent.columns = ["Date","Customer","Total (₹)","Discount Rule"]
            st.dataframe(recent, use_container_width=True, hide_index=True)

        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    if st.session_state.role is None:
        login_page()
        return

    with st.sidebar:
        name = st.session_state.username or st.session_state.owner_name
        st.markdown(f"**{name}**")
        st.markdown(
            f'<span class="bdg bdg-blue">'
            f'{"Customer" if st.session_state.role=="customer" else "Owner"}</span>',
            unsafe_allow_html=True
        )
        st.markdown("")
        if st.button("🚪 Logout", use_container_width=True):
            for k, v in _defaults.items():
                st.session_state[k] = v
            st.rerun()

    if st.session_state.role == "customer":
        customer_dashboard()
    elif st.session_state.role == "owner":
        owner_dashboard()

main()
