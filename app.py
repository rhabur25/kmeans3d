import streamlit as st
import numpy as np
import plotly.graph_objs as go
from sklearn.cluster import KMeans

st.set_page_config(page_title="3D K-Means Clustering", layout="wide")
st.title("3D K-Means Clustering Visualizer")

# Sidebar controls
st.sidebar.header("Controls")
num_points = st.sidebar.slider("Number of Points", 10, 500, 100)
num_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)

if 'step' not in st.session_state:
    st.session_state.step = 0
if 'points' not in st.session_state:
    st.session_state.points = None
if 'labels' not in st.session_state:
    st.session_state.labels = None
if 'centers' not in st.session_state:
    st.session_state.centers = None
if 'kmeans' not in st.session_state:
    st.session_state.kmeans = None

# Generate random points
def generate_points():
    st.session_state.points = np.random.rand(num_points, 3)
    st.session_state.labels = np.zeros(num_points, dtype=int)
    st.session_state.centers = None
    st.session_state.kmeans = KMeans(n_clusters=num_clusters, n_init=1, max_iter=1, init='random', random_state=42)
    st.session_state.step = 0

gen_btn = st.sidebar.button("Generate Points", on_click=generate_points)

# Step clustering
step_btn = st.sidebar.button("Next Step")
reset_btn = st.sidebar.button("Reset Steps", on_click=lambda: st.session_state.update({'step': 0}))

if st.session_state.points is not None:
    if step_btn:
        if st.session_state.kmeans is None or st.session_state.kmeans.n_clusters != num_clusters:
            st.session_state.kmeans = KMeans(n_clusters=num_clusters, n_init=1, max_iter=1, init='random', random_state=42)
        # Fit for one more step
        kmeans = KMeans(n_clusters=num_clusters, n_init=1, max_iter=st.session_state.step+1, init=st.session_state.kmeans.cluster_centers_ if st.session_state.centers is not None else 'random', random_state=42)
        kmeans.fit(st.session_state.points)
        st.session_state.labels = kmeans.labels_
        st.session_state.centers = kmeans.cluster_centers_
        st.session_state.kmeans = kmeans
        st.session_state.step += 1

    # Plot
    fig = go.Figure()
    for i in range(num_clusters):
        mask = st.session_state.labels == i
        fig.add_trace(go.Scatter3d(
            x=st.session_state.points[mask,0],
            y=st.session_state.points[mask,1],
            z=st.session_state.points[mask,2],
            mode='markers',
            marker=dict(size=5),
            name=f'Cluster {i+1}'
        ))
    if st.session_state.centers is not None:
        fig.add_trace(go.Scatter3d(
            x=st.session_state.centers[:,0],
            y=st.session_state.centers[:,1],
            z=st.session_state.centers[:,2],
            mode='markers',
            marker=dict(size=15, color='black', symbol='x'),
            name='Centers'
        ))
    fig.update_layout(
        width=900, height=700,
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Click 'Generate Points' to start.")

st.markdown("""
- Use the sidebar to set the number of points and clusters.
- Click 'Generate Points' to create a new dataset.
- Click 'Next Step' to advance the clustering one iteration at a time.
- You can rotate and zoom the 3D plot interactively.
""")
