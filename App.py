import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AI Mall Dashboard", layout="wide")

st.title("🛍️ AI-Powered Mall Customer Segmentation Dashboard")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")

df = load_data()

# -------------------------------
# SIDEBAR FILTERS
# -------------------------------
st.sidebar.header("🔍 Filters")

gender = st.sidebar.multiselect("Gender", df["Gender"].unique(), default=df["Gender"].unique())

age = st.sidebar.slider("Age", int(df["Age"].min()), int(df["Age"].max()), (18, 60))

filtered_df = df[
    (df["Gender"].isin(gender)) &
    (df["Age"].between(age[0], age[1]))
]

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3 = st.tabs(["📊 EDA", "🤖 Clustering", "💡 Insights"])

# -------------------------------
# TAB 1: EDA
# -------------------------------
with tab1:
    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Customers", len(filtered_df))
    col2.metric("Avg Income", round(filtered_df["Annual Income (k$)"].mean(), 1))
    col3.metric("Avg Spending", round(filtered_df["Spending Score (1-100)"].mean(), 1))

    st.dataframe(filtered_df.head())

    # Charts
    st.subheader("Visualizations")

    fig1 = px.histogram(filtered_df, x="Age", title="Age Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(filtered_df,
                      x="Annual Income (k$)",
                      y="Spending Score (1-100)",
                      color="Gender",
                      title="Income vs Spending")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.pie(filtered_df, names="Gender", title="Gender Distribution")
    st.plotly_chart(fig3, use_container_width=True)

# -------------------------------
# TAB 2: CLUSTERING
# -------------------------------
with tab2:
    st.subheader("K-Means Clustering")

    X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

    # Elbow Method
    st.write("### Elbow Method")

    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    elbow_df = pd.DataFrame({"Clusters": range(1, 11), "WCSS": wcss})
    fig_elbow = px.line(elbow_df, x="Clusters", y="WCSS", title="Elbow Method")
    st.plotly_chart(fig_elbow, use_container_width=True)

    # Clustering
    k = st.slider("Select Number of Clusters", 2, 10, 5)

    kmeans = KMeans(n_clusters=k, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X)

    fig_cluster = px.scatter(
        df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        color=df["Cluster"].astype(str),
        title="Customer Segments"
    )

    st.plotly_chart(fig_cluster, use_container_width=True)

# -------------------------------
# TAB 3: INSIGHTS
# -------------------------------
with tab3:
    st.subheader("Business Insights")

    st.write("""
    • High-income low-spending customers are potential targets for marketing campaigns.  
    • Young customers tend to have higher spending scores.  
    • Clustering helps identify distinct behavioral groups.  
    • Businesses can use these insights for personalized offers and retention strategies.
    """)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("Developed using Streamlit, Plotly & Machine Learning 🚀")