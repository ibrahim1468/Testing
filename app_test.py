import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

st.title("DSML Environment Compatibility Test ✅")
st.write("Testing numpy, pandas, matplotlib, seaborn, scipy, scikit-learn, and Streamlit together")

# --- NumPy ---
st.subheader("NumPy Array")
arr = np.random.randint(1, 101, size=(10,))
st.write("Random NumPy array:", arr)

# --- Pandas ---
st.subheader("Pandas DataFrame")
df = pd.DataFrame({
    'A': np.random.randn(50),
    'B': np.random.randn(50)
})
st.write(df.head())

# --- Matplotlib ---
st.subheader("Matplotlib Plot")
fig, ax = plt.subplots()
ax.plot(df['A'], df['B'], 'o-', color='purple', label='A vs B')
ax.set_xlabel("A")
ax.set_ylabel("B")
ax.set_title("Matplotlib Scatter Line Plot")
ax.legend()
st.pyplot(fig)

# --- Seaborn ---
st.subheader("Seaborn Plot")
fig2, ax2 = plt.subplots()
sns.histplot(df['A'], kde=True, ax=ax2)
ax2.set_title("Seaborn Histogram with KDE")
st.pyplot(fig2)

# --- Scipy ---
st.subheader("Scipy Stats")
slope, intercept, r_value, p_value, std_err = stats.linregress(df['A'], df['B'])
st.write(f"Linear regression (scipy): slope={slope:.3f}, intercept={intercept:.3f}, r={r_value:.3f}")

# --- Sklearn ---
st.subheader("Scikit-learn Linear Regression")
X = df[['A']]
y = df['B']
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
st.write(f"Linear regression (sklearn): coef={model.coef_[0]:.3f}, intercept={model.intercept_:.3f}")

# --- Done ---
st.success("All libraries are working together smoothly! 🎉")