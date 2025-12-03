import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

def compute_alpha_beta(user_ai_votes, user_not_votes, user_acc_ai, user_acc_real):
    userAi        = user_acc_ai                 * user_ai_votes
    userAiNot     = (1 - user_acc_real)         * user_not_votes
    userReal      = user_acc_real               * user_not_votes
    userRealNot   = (1 - user_acc_ai)           * user_ai_votes

    alpha = 1 + userAi + userAiNot
    beta  = 1 + userReal + userRealNot
    return alpha, beta

def prob_ai(alpha, beta_val):
    p_real = beta_dist.cdf(0.5, alpha, beta_val)
    return 1 - p_real

st.title("Bayesian Decision Heatmap (Interactive Votes)")

col1, col2 = st.columns(2)

with col1:
    user_acc_ai = st.slider("User Accuracy — AI", 0.0, 1.0, 0.6, 0.01)
with col2:
    user_acc_real = st.slider("User Accuracy — Real", 0.0, 1.0, 0.8, 0.01)

threshold = st.slider("Threshold", 0.0, 1.0, 0.9, 0.01)
st.divider()
st.subheader("Select Vote Combination")

v1, v2 = st.columns(2)

with v1:
    current_ai = st.slider("AI Votes", 0, 50, 10)
with v2:
    current_real = st.slider("Real Votes", 0, 50, 5)

st.divider()


max_votes = 50
ai_range = np.arange(0, max_votes + 1)
real_range = np.arange(0, max_votes + 1)

Z = np.zeros((len(real_range), len(ai_range)))

for i, rv in enumerate(real_range):
    for j, av in enumerate(ai_range):
        a, b = compute_alpha_beta(av, rv, user_acc_ai, user_acc_real)
        Z[i, j] = prob_ai(a, b)

alpha_sel, beta_sel = compute_alpha_beta(current_ai, current_real, user_acc_ai, user_acc_real)
prob_sel = prob_ai(alpha_sel, beta_sel)

fig, ax = plt.subplots(figsize=(10, 8))

cmap = plt.cm.coolwarm
cax = ax.imshow(
    Z, origin='lower', cmap=cmap,
    extent=[0, max_votes, 0, max_votes], vmin=0, vmax=1
)


cs = ax.contour(
    ai_range, real_range, Z,
    levels=[threshold], colors='yellow', linewidths=2
)
ax.clabel(cs, inline=True, fontsize=10)


ax.scatter(current_ai, current_real, s=200, c="black", edgecolors="white", linewidth=2, label="Selected Votes")

# Text label
ax.text(current_ai + 1, current_real + 1, f"P(AI)={prob_sel:.2f}", color="white", fontsize=12)

ax.set_title("Decision Heatmap (AI Votes vs Real Votes)")
ax.set_xlabel("AI Votes")
ax.set_ylabel("Real Votes")

fig.colorbar(cax, label="P(AI)")

st.pyplot(fig)

st.markdown(f"### Selected Vote Confidence: **{prob_sel:.3f}**")
st.markdown(f"### Above threshold: **{prob_sel >= threshold}**")
