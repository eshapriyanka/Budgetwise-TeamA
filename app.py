# imports
import streamlit as st
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns
import io
import time
import random
import nltk
import sqlite3
import hashlib
from prophet import Prophet
import altair as alt

# NLTK Setup
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))

# DATABASE CONFIGURATION
DB_FILE = 'budgetwise_pro.db'

def init_db():
    """Initialize the database and seed default data."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Create Tables
    c.execute('''CREATE TABLE IF NOT EXISTS Users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password_hash TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS Admins (
        admin_id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password_hash TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS Categories (
        category_id INTEGER PRIMARY KEY AUTOINCREMENT, category_name TEXT UNIQUE, keywords TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS Transactions (
        transaction_id TEXT PRIMARY KEY, user_id INTEGER, date TEXT, type TEXT, amount REAL, 
        description TEXT, category_name TEXT, FOREIGN KEY(user_id) REFERENCES Users(user_id))''')
    c.execute('''CREATE TABLE IF NOT EXISTS Goals (
        goal_id TEXT PRIMARY KEY, user_id INTEGER, category_name TEXT, target_amount REAL, 
        FOREIGN KEY(user_id) REFERENCES Users(user_id))''')
    
    # Seed Admin
    c.execute("SELECT * FROM Admins WHERE username = 'admin'")
    if not c.fetchone():
        c.execute("INSERT INTO Admins (username, password_hash) VALUES (?, ?)", 
                  ('admin', hashlib.sha256('admin123'.encode()).hexdigest()))
    
    # FULL CATEGORIES
    c.execute("SELECT count(*) FROM Categories")
    if c.fetchone()[0] == 0:
        defaults = {
            'Housing': ['rent', 'mortgage', 'property', 'tax', 'home', 'insurance', 'hoa', 'plumbing', 'electrician', 'repair', 'furniture', 'ikea', 'depot'],
            'Transportation': ['uber', 'car', 'lyft', 'taxi', 'bus', 'subway', 'amtrak', 'train', 'fuel', 'gasoline', 'payment', 'insurance', 'parking', 'wash', 'auto', 'repair', 'dmv', 'bolt'],
            'Groceries & Household': ['grocery', 'groceries', 'market', 'safeway', 'kroger', 'walmart', 'costco', 'sprouts', 'trader', 'joe', 'publix', 'food', 'supermarket', 'target', 'whole', 'foods', 'household', 'supplies', 'toilet', 'paper', 'soap', 'detergent'],
            'Dining': ['restaurant', 'cafe', 'coffee', 'snaks', 'starbucks', 'doordash', 'grubhub', 'ubereats', 'delivery', 'mcdonalds', 'burger', 'king', 'pizza', 'hut', 'dominos', 'chipotle', 'eats', 'eating', 'out'],
            'Entertainment': ['movie', 'cinema', 'concert', 'spotify', 'netflix', 'hulu', 'disney', 'app', 'store', 'google', 'play', 'tickets', 'bar', 'nightclub', 'apple', 'music', 'youtube', 'premium', 'gaming', 'steam', 'playstation', 'xbox'],
            'Personal & Family Care': ['haircut', 'shopping' ,'salon', 'barber', 'cosmetics', 'toiletries', 'sephora', 'ulta', 'gym', 'fitness', 'yoga', 'childcare', 'daycare', 'baby', 'pet', 'food', 'vet', 'veterinarian', 'spa', 'massage'],
            'Work & Education': ['office', 'supplies', 'stationery', 'udemy', 'coursera', 'book', 'textbook', 'tuition', 'school', 'college', 'webinar', 'software', 'adobe', 'slack', 'zoom', 'linkedin', 'learning', 'github'],
            'Health & Medical': ['doctor', 'dentist', 'hospital', 'pharmacy', 'cvs', 'walgreens', 'rite', 'aid', 'medicine', 'prescription', 'health', 'insurance', 'copay', 'vision', 'therapy', 'physician'],
            'Travel': ['flight', 'airline', 'american', 'delta', 'united', 'southwest', 'hotel', 'airbnb', 'booking.com', 'expedia', 'vacation', 'trip', 'luggage', 'rental', 'car', 'hertz', 'avis'],
            'Technology & Communication': ['phone', 'bill', 'verizon', 'at&t', 't-mobile', 'internet', 'comcast', 'xfinity', 'google', 'fi', 'gadget', 'apple', 'samsung', 'best', 'buy', 'newegg', 'aws', 'gcp', 'azure', 'domain'],
            'Financial & Insurance': ['life', 'insurance', 'bank', 'fee', 'atm', 'financial', 'advisor', 'investment', 'stock', 'coinbase', 'robinhood', 'loan', 'payment', 'student', 'credit', 'card', 'transfer'],
            'Business Expenses': ['client', 'dinner', 'business', 'travel', 'consulting', 'legal', 'fee', 'advertising', 'quickbooks', 'adwords'],
            'Taxes': ['tax', 'return', 'irs', 'property', 'income', 'prep', 'h&r', 'block', 'turbotax'],
            'Income': ['salary', 'paycheck', 'deposit', 'bonus', 'freelance', 'invoice', 'refund', 'reimbursement', 'interest', 'dividend'],
            'Other': ['charity', 'donation', 'gift']
        }
        for cat, keys in defaults.items():
            c.execute("INSERT INTO Categories (category_name, keywords) VALUES (?, ?)", (cat, ",".join(keys)))
    
    conn.commit()
    conn.close()

init_db()

# HELPER FUNCTIONS
def hash_pass(p): return hashlib.sha256(p.encode()).hexdigest()

def get_categories_dict():
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("SELECT category_name, keywords FROM Categories")
    data = {row[0]: set(row[1].split(",")) if row[1] else set() for row in c.fetchall()}
    conn.close(); return data

def categorize_transaction_nltk(desc, t_type, cats):
    if not isinstance(desc, str) or not desc.strip(): return 'Other'
    tokens = {w for w in word_tokenize(desc.lower()) if w.isalpha() and w not in stop_words}
    if not tokens: return 'Other'
    if t_type == "Income": return 'Income' if 'Income' in cats and not tokens.isdisjoint(cats['Income']) else 'Income'
    for c, k in cats.items():
        if c == 'Income': continue
        if not tokens.isdisjoint(k): return c
    return 'Other'

# DB OPERATIONS
def get_user_transactions(uid):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM Transactions WHERE user_id=?", conn, params=(uid,))
    conn.close()
    if not df.empty:
        df = df.rename(columns={'transaction_id':'id', 'date':'Date', 'type':'Type', 'amount':'Amount', 'description':'Description', 'category_name':'Category'})
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df

def save_transactions_db(df):
    conn = sqlite3.connect(DB_FILE)
    if not df.empty:
        df = df.copy()
        # Ensure Date is string YYYY-MM-DD for SQLite
        df['Date'] = df['Date'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) and not isinstance(x, str) else x)
    data = list(df.itertuples(index=False, name=None))
    conn.executemany('INSERT OR REPLACE INTO Transactions VALUES (?,?,?,?,?,?,?)', data)
    conn.commit(); conn.close()

def update_transactions_bulk(df_to_update):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    data = []
    for _, row in df_to_update.iterrows():
        d_str = row['Date'].strftime('%Y-%m-%d') if pd.notnull(row['Date']) and not isinstance(row['Date'], str) else row['Date']
        data.append((d_str, row['Type'], row['Amount'], row['Description'], row['Category'], row['user_id'], row['id']))
    c.executemany('UPDATE Transactions SET date=?, type=?, amount=?, description=?, category_name=?, user_id=? WHERE transaction_id=?', data)
    conn.commit(); conn.close()

def delete_transaction_ids(id_list):
    if not id_list: return
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    placeholders = ','.join('?' for _ in id_list)
    c.execute(f"DELETE FROM Transactions WHERE transaction_id IN ({placeholders})", id_list)
    conn.commit(); conn.close()

def delete_all_user_transactions(uid):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("DELETE FROM Transactions WHERE user_id=?", (uid,))
    conn.commit(); conn.close()

def save_goal_db(uid, cat, amt):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("SELECT goal_id FROM Goals WHERE user_id=? AND category_name=?", (uid, cat))
    res = c.fetchone()
    if res: c.execute("UPDATE Goals SET target_amount=? WHERE goal_id=?", (amt, res[0]))
    else: c.execute("INSERT INTO Goals VALUES (?,?,?,?)", (f"goal_{uid}_{time.time()}", uid, cat, amt))
    conn.commit(); conn.close()

def get_user_goals(uid):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM Goals WHERE user_id=?", conn, params=(uid,))
    conn.close()
    return df

def delete_goal_db(goal_id):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor(); c.execute("DELETE FROM Goals WHERE goal_id=?", (goal_id,)); conn.commit(); conn.close()

# PROCESSING
def process_uploaded_data(df_raw, uid, cats):
    df = df_raw.copy()
    d_col = next((c for c in df.columns if c.lower() == 'date'), None)
    if not d_col: st.error("Missing 'Date'"); return None
    df.dropna(subset=[d_col], inplace=True)
    col_map = {d_col: 'Date'}
    for c in df.columns:
        if c.lower() == 'amount': col_map[c] = 'Amount'
        if c.lower() == 'description': col_map[c] = 'Description'
        if c.lower() == 'type': col_map[c] = 'Type'
    df.rename(columns=col_map, inplace=True)
    try:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').abs()
    except: return None
    if 'Description' not in df.columns: df['Description'] = "Uploaded"
    df['Description'] = df['Description'].fillna("").astype(str)
    if 'Type' not in df.columns: df['Type'] = 'Expense'
    
    df['Category'] = df.apply(lambda x: categorize_transaction_nltk(x['Description'], x['Type'], cats), axis=1)
    df['transaction_id'] = [f"{uid}_{time.time()}_{i}" for i in range(len(df))]
    df['user_id'] = uid
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    return df[['transaction_id', 'user_id', 'Date', 'Type', 'Amount', 'Description', 'Category']]

# UI
st.set_page_config(page_title="BudgetWise", layout="wide")

# Session State Init
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'uploaded_file_processed' not in st.session_state: st.session_state.uploaded_file_processed = False
if 'current_uploaded_file' not in st.session_state: st.session_state.current_uploaded_file = None
if 'invalid_date_count' not in st.session_state: st.session_state.invalid_date_count = 0

# LOGIN PAGE
if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align: center; color: #0d47a1;'>Welcome to BudgetWise</h1>", unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["User Login", "Register", "Admin Login"])
    with t1:
        with st.form("ul"):
            u = st.text_input("Username"); p = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                conn = sqlite3.connect(DB_FILE); c = conn.cursor()
                c.execute("SELECT user_id FROM Users WHERE username=? AND password_hash=?", (u, hash_pass(p)))
                res = c.fetchone(); conn.close()
                if res:
                    st.session_state.logged_in = True; st.session_state.user_id = res[0]
                    st.session_state.username = u; st.session_state.role = 'user'; st.rerun()
                else: st.error("Invalid")
    with t2:
        with st.form("reg"):
            nu = st.text_input("New User"); np = st.text_input("New Pass", type="password")
            if st.form_submit_button("Register"):
                try:
                    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
                    c.execute("INSERT INTO Users (username, password_hash) VALUES (?,?)", (nu, hash_pass(np)))
                    conn.commit(); conn.close(); st.success("Created! Login now.")
                except: st.error("Username exists")
    with t3:
        with st.form("al"):
            au = st.text_input("Admin User"); ap = st.text_input("Admin Pass", type="password")
            if st.form_submit_button("Login"):
                conn = sqlite3.connect(DB_FILE); c = conn.cursor()
                c.execute("SELECT admin_id FROM Admins WHERE username=? AND password_hash=?", (au, hash_pass(ap)))
                res = c.fetchone(); conn.close()
                if res:
                    st.session_state.logged_in = True; st.session_state.username = au; st.session_state.role = 'admin'; st.rerun()
                else: st.error("Invalid Admin")

# DASHBOARD
else:
    # Load Category Dict Once
    CATEGORIES_KEYWORDS = get_categories_dict()

    # ADMIN DASHBOARD
    if st.session_state.role == 'admin':
        c1, c2 = st.columns([6, 1])
        c1.title("Admin Dashboard")
        if c2.button("Logout"):
            st.session_state.clear(); st.rerun()
        
        conn = sqlite3.connect(DB_FILE)
        
        # System Stats
        st.subheader("System Stats")
        u_count = pd.read_sql("SELECT count(*) FROM Users", conn).iloc[0,0]
        t_count = pd.read_sql("SELECT count(*) FROM Transactions", conn).iloc[0,0]
        st.write(f"**Total Users:** {u_count} | **Total Transactions:** {t_count}")
        
        st.markdown("---")
        
        # User Drill-Down
        st.subheader("User Management")
        users = pd.read_sql("SELECT user_id, username FROM Users", conn)
        user_map = {row['username']: row['user_id'] for _, row in users.iterrows()}
        
        selected_user_name = st.selectbox("Select User to View Details", ["Select User"] + list(user_map.keys()))
        
        if selected_user_name != "Select User":
            sel_uid = user_map[selected_user_name]
            
            # Get Data for Selected User
            user_trans = get_user_transactions(sel_uid)
            user_goals = get_user_goals(sel_uid)
            
            st.write(f"### Data for: {selected_user_name}")
            
            # Show Goals
            with st.expander("User Goals"):
                if not user_goals.empty:
                    st.dataframe(user_goals, use_container_width=True)
                else:
                    st.info("No goals set.")
            
            # Show Transactions
            with st.expander("User Transactions", expanded=True):
                if not user_trans.empty:
                    st.dataframe(user_trans, use_container_width=True)
                else:
                    st.info("No transactions found.")

        conn.close()
        
        st.markdown("---")
        
        # Category Management
        st.subheader("Category Management")
        c1, c2 = st.columns(2)
        with c1:
            with st.form("add_cat"):
                nc = st.text_input("New Category Name")
                if st.form_submit_button("Create Category"):
                    try:
                        conn = sqlite3.connect(DB_FILE)
                        conn.execute("INSERT INTO Categories (category_name, keywords) VALUES (?,?)", (nc, ""))
                        conn.commit(); conn.close(); st.success(f"Category '{nc}' created!"); st.rerun()
                    except Exception as e: st.error(f"Error: {e}")
        with c2:
            with st.form("add_kw"):
                tc = st.selectbox("Select Category", list(CATEGORIES_KEYWORDS.keys()))
                kw = st.text_input("New Keyword (lowercase)")
                if st.form_submit_button("Add Keyword"):
                    if tc and kw.strip().isalpha():
                        conn = sqlite3.connect(DB_FILE)
                        curr = conn.execute("SELECT keywords FROM Categories WHERE category_name=?", (tc,)).fetchone()[0]
                        nkw = (curr + "," + kw.lower()) if curr else kw.lower()
                        conn.execute("UPDATE Categories SET keywords=? WHERE category_name=?", (nkw, tc))
                        conn.commit(); conn.close(); st.success(f"Added '{kw}' to {tc}"); st.rerun()
                    else: st.error("Invalid keyword.")
        
        st.write("**Current Categories:**")
        for category, keywords in CATEGORIES_KEYWORDS.items():
            with st.expander(f"{category}"):
                st.write(", ".join(sorted(list(keywords))))

    # USER DASHBOARD
    else:
        c1, c2 = st.columns([6, 1])
        c1.title(f"Welcome, {st.session_state.username}")
        if c2.button("Logout"):
            st.session_state.clear(); st.rerun()
        
        # Load User Data
        df = get_user_transactions(st.session_state.user_id)
        expenses_df = pd.DataFrame()
        if not df.empty:
            expenses_df = df[df['Type'] == 'Expense'].copy()
        
        t1, t2, t3, t4 = st.tabs(["Log & Manage", "Analysis", "Goals", "Forecast"])
        
        # TAB 1: Manage
        with t1:
            c1, c2 = st.columns(2)
            with c1:
                with st.expander("Upload CSV", expanded=True):
                    uploaded_file = st.file_uploader("Choose a file", type="csv", key='file_uploader_widget')
                    if uploaded_file is not None:
                        if uploaded_file != st.session_state.current_uploaded_file or not st.session_state.uploaded_file_processed:
                            st.session_state.current_uploaded_file = uploaded_file
                            st.session_state.invalid_date_count = 0 
                            try:
                                with st.spinner('Processing...'):
                                    df_read = pd.read_csv(uploaded_file, low_memory=False)
                                    new_data = process_uploaded_data(df_read, st.session_state.user_id, CATEGORIES_KEYWORDS)
                                if new_data is not None:
                                    save_transactions_db(new_data)
                                    st.session_state.uploaded_file_processed = True
                                    st.success(f"Saved {len(new_data)} transactions to DB!")
                                    if st.session_state.invalid_date_count > 0: st.warning(f"{st.session_state.invalid_date_count} rows removed (bad dates).")
                                    st.rerun() 
                                else:
                                    st.session_state.current_uploaded_file = None; st.session_state.uploaded_file_processed = False
                            except Exception as e:
                                st.error(f"Upload error: {e}"); st.session_state.current_uploaded_file = None; st.session_state.uploaded_file_processed = False
                    elif st.session_state.current_uploaded_file is not None:
                         st.info("File removed."); st.session_state.current_uploaded_file = None; st.session_state.uploaded_file_processed = False; st.session_state.invalid_date_count = 0; st.rerun()

            with c2:
                with st.expander("Manual Entry", expanded=True):
                    with st.form("man"):
                        dt = st.date_input("Date", date.today()); am = st.number_input("Amount", min_value=0.01)
                        ty = st.selectbox("Type", ["Expense", "Income"]); de = st.text_input("Desc")
                        if st.form_submit_button("Add"):
                            cat = categorize_transaction_nltk(de, ty, CATEGORIES_KEYWORDS)
                            tid = f"{st.session_state.user_id}_{time.time()}"
                            new_t = pd.DataFrame([[tid, st.session_state.user_id, dt, ty, am, de, cat]], 
                                                columns=['transaction_id', 'user_id', 'Date', 'Type', 'Amount', 'Description', 'Category'])
                            new_t['Date'] = new_t['Date'].astype(str)
                            save_transactions_db(new_t)
                            st.success("Added"); st.rerun()
            
            st.divider()
            st.subheader("Edit Transactions")
            
            if not df.empty:
                with st.expander("⚠️ Danger Zone"):
                     if st.button("Delete ALL Transactions", type="primary"):
                          delete_all_user_transactions(st.session_state.user_id)
                          st.success("All data cleared."); time.sleep(1); st.rerun()
                
                ROWS_PER_PAGE = 50
                total_rows = len(df)
                df_sorted = df.sort_values(by='Date', ascending=False).reset_index(drop=True)
                
                if total_rows > 500:
                    st.info(f"Showing {ROWS_PER_PAGE} of {total_rows} transactions.")
                    page = st.slider("Page", 1, (total_rows // ROWS_PER_PAGE) + 1, 1)
                    start = (page - 1) * ROWS_PER_PAGE
                    end = start + ROWS_PER_PAGE
                    editor_data = df_sorted.iloc[start:end]
                    is_paginated = True
                else:
                    editor_data = df_sorted
                    is_paginated = False
                
                edited = st.data_editor(
                    editor_data, 
                    num_rows="dynamic",
                    column_config={
                        "id": None, "user_id": None,
                        "Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                        "Amount": st.column_config.NumberColumn(format="$%.2f"),
                        "Category": st.column_config.SelectboxColumn(options=list(CATEGORIES_KEYWORDS.keys())),
                        "Type": st.column_config.SelectboxColumn(options=["Expense", "Income"])
                    },
                    key="editor"
                )

                if st.button("Save Changes"):
                    if not edited.equals(editor_data):
                        edited['user_id'] = st.session_state.user_id
                        update_transactions_bulk(edited)
                        
                        if not is_paginated:
                            old_ids = set(editor_data['id']); new_ids = set(edited['id'])
                            deleted = old_ids - new_ids
                            if deleted: delete_transaction_ids(list(deleted))
                            
                        st.success("Updated DB!"); st.rerun()
            else: st.info("No data.")
        
        # TAB 2: Analysis
        with t2:
            if not df.empty:
                inc = df[df['Type']=='Income']['Amount'].sum()
                exp = expenses_df['Amount'].sum() if not expenses_df.empty else 0
                c1,c2,c3 = st.columns(3)
                c1.metric("Income", f"${inc:,.2f}"); c2.metric("Expense", f"${exp:,.2f}"); c3.metric("Savings", f"${inc-exp:,.2f}")
                
                if not expenses_df.empty:
                    c1, c2 = st.columns(2)
                    with c1: 
                        st.write("**Category Breakdown**")
                        grp = expenses_df.groupby('Category')['Amount'].sum()
                        fig, ax = plt.subplots()
                        ax.pie(grp, labels=grp.index, autopct='%1.1f%%')
                        st.pyplot(fig)
                    with c2:
                        st.write("**Trend**")
                        daily = expenses_df.set_index('Date').resample('ME')['Amount'].sum()
                        st.line_chart(daily)
        
        # TAB 3: Goals
        with t3:
            goals_df = get_user_goals(st.session_state.user_id)
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Set Goal**")
                cats = [c for c in CATEGORIES_KEYWORDS.keys() if c != 'Income']
                g_cat = st.selectbox("Category", cats if cats else ["None"])
                g_val = st.number_input("Limit", min_value=1.0)
                if st.button("Save Goal"):
                    save_goal_db(st.session_state.user_id, g_cat, g_val); st.rerun()
            
            with c2:
                st.write("**Progress (This Month)**")
                if not goals_df.empty and not expenses_df.empty:
                    curr = expenses_df[expenses_df['Date'].dt.to_period('M') == pd.Timestamp.now().to_period('M')]
                    spend_map = curr.groupby('Category')['Amount'].sum()
                    
                    for _, g in goals_df.iterrows():
                        cat = g['category_name']
                        tgt = g['target_amount']
                        s = spend_map.get(cat, 0)
                        
                        st.write(f"**{cat}**")
                        st.caption(f"${s:,.0f} / ${tgt:,.0f}")
                        
                        ratio = s / tgt if tgt > 0 else 0
                        if ratio > 1.0:
                            st.progress(1.0)
                            st.error(f"Over Budget by ${s-tgt:,.0f}!")
                        elif ratio > 0.9:
                            st.progress(ratio)
                            st.warning("Near Limit")
                        else:
                            st.progress(ratio)
                        
                        if st.button(f"Delete {cat}", key=g['goal_id']):
                            delete_goal_db(g['goal_id']); st.rerun()
                else: st.info("No goals or data.")

        # TAB 4: Forecast
        with t4:
            if not expenses_df.empty and len(expenses_df) > 30:
                try:
                    daily_expenses = expenses_df.set_index('Date').resample('D')['Amount'].sum().reset_index()
                    daily_expenses.rename(columns={'Date': 'ds', 'Amount': 'y'}, inplace=True)
                    
                    if len(daily_expenses.index) < 30:
                         st.info("Not enough data (need >30 days) for forecasting.")
                    else:
                        col_opt, col_days = st.columns([1, 1])
                        with col_opt:
                            history_view = st.selectbox("Select Historical Data View:", 
                                                        ['Daily Total (All Data)', 'Weekly Average (Recommended)', 'Monthly Average', 'Yearly Average'])
                        with col_days:
                            forecast_days = st.slider("Select days to forecast:", 30, 365, 90)
    
                        if st.button("Generate Forecast", use_container_width=True):
                            with st.spinner(f"Training model and forecasting {forecast_days} days..."):
                                m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True, changepoint_prior_scale=0.05)
                                m.fit(daily_expenses)
                                future = m.make_future_dataframe(periods=forecast_days)
                                forecast = m.predict(future)
                                
                                st.subheader(f"Forecasted Expenses for the Next {forecast_days} Days")
                                
                                # Chart Prep
                                if history_view == 'Daily Total (All Data)':
                                    actuals_plot = daily_expenses.rename(columns={'y': 'Actual'})
                                    t_fmt = '%Y-%m-%d'; lbl = "Actual Daily"
                                elif history_view == 'Weekly Average (Recommended)':
                                    actuals_plot = daily_expenses.set_index('ds').resample('W')['y'].mean().reset_index().rename(columns={'y': 'Actual'})
                                    t_fmt = 'Week of %Y-%m-%d'; lbl = "Actual Weekly"
                                elif history_view == 'Monthly Average':
                                    actuals_plot = daily_expenses.set_index('ds').resample('ME')['y'].mean().reset_index().rename(columns={'y': 'Actual'})
                                    t_fmt = '%Y-%m'; lbl = "Actual Monthly"
                                else:
                                    actuals_plot = daily_expenses.set_index('ds').resample('YE')['y'].mean().reset_index().rename(columns={'y': 'Actual'})
                                    t_fmt = '%Y'; lbl = "Actual Yearly"

                                f_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                                
                                base = alt.Chart(f_data).encode(x='ds:T')
                                line = base.mark_line(color='#0d47a1').encode(y='yhat')
                                band = base.mark_area(opacity=0.3, color='#42a5f5').encode(y='yhat_lower', y2='yhat_upper')
                                points = alt.Chart(actuals_plot).mark_circle(color='black').encode(x='ds:T', y='Actual', tooltip=[alt.Tooltip('ds:T', format=t_fmt), 'Actual'])
                                
                                st.markdown(f"**Legend:** Black={lbl}, Blue=Prediction, Area=Confidence")
                                st.altair_chart((band + line + points).interactive(), use_container_width=True)
                                
                                st.subheader("Trends")
                                last_hist = daily_expenses['ds'].max()
                                future_comp = forecast[forecast['ds'] > last_hist][['ds', 'trend', 'weekly', 'yearly']].copy()
                                comp = future_comp.melt('ds')
                                c_chart = alt.Chart(comp).mark_line().encode(x='ds:T', y='value', color='variable').facet(row='variable').resolve_scale(y='independent')
                                st.altair_chart(c_chart, use_container_width=True)
                                
                                st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days).set_index('ds'))

                except Exception as e: st.error(f"Forecasting error: {e}")
            else: st.info("Need >30 days data")