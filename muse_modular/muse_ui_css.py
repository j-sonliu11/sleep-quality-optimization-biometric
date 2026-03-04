import streamlit as st

def inject_css():
    st.markdown(
        """
<style>
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
section[data-testid="stSidebar"] > div { padding-top: 0.35rem; padding-bottom: 0.8rem; }
div.stButton > button { border-radius: 14px; padding: 0.75rem 1rem; }
.element-container { margin-bottom: 0.4rem; }

.run-badge { display:inline-flex; align-items:center; gap:10px; }
.spinner {
  width:14px; height:14px;
  border:2px solid rgba(255,255,255,0.25);
  border-top-color: rgba(255,255,255,0.85);
  border-radius:50%;
  animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.small-muted { color: rgba(255,255,255,0.7); font-size: 0.95rem; }

div[data-testid="stStatusWidget"] { display: none !important; }

div[data-testid="stHorizontalBlock"] { align-items: flex-start !important; }

div[data-testid="stHorizontalBlock"],
div[data-testid="column"],
div[data-testid="stVerticalBlock"],
div[data-testid="stMainBlockContainer"],
div[data-testid="stBlock"],
div[data-testid="stElementContainer"],
div.block-container {
  overflow: visible !important;
}

#dc-row-anchor { display: none; }

div[data-testid="stElementContainer"]:has(#dc-row-anchor)
  + div[data-testid="stHorizontalBlock"] {
  align-items: flex-start !important;
}

div[data-testid="stElementContainer"]:has(#dc-row-anchor)
  + div[data-testid="stHorizontalBlock"]
  > div[data-testid="column"]:nth-of-type(2) {
  position: sticky !important;
  top: 0.85rem !important;
  align-self: flex-start !important;
  z-index: 40 !important;
}

div[data-testid="stElementContainer"]:has(#dc-row-anchor)
  + div[data-testid="stHorizontalBlock"]
  > div[data-testid="column"]:nth-of-type(2)
  > div[data-testid="stVerticalBlock"] {
  background: rgba(17, 24, 39, 0.72);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 0.85rem 0.9rem 0.9rem 0.9rem;
  box-shadow: 0 8px 20px rgba(0,0,0,0.22);
  max-height: calc(100vh - 1.7rem);
  overflow: auto !important;
}

@media (max-width: 899px) {
  div[data-testid="stElementContainer"]:has(#dc-row-anchor)
    + div[data-testid="stHorizontalBlock"]
    > div[data-testid="column"]:nth-of-type(2) {
    position: static !important;
    top: auto !important;
    z-index: auto !important;
  }
  div[data-testid="stElementContainer"]:has(#dc-row-anchor)
    + div[data-testid="stHorizontalBlock"]
    > div[data-testid="column"]:nth-of-type(2)
    > div[data-testid="stVerticalBlock"] {
    max-height: none !important;
    overflow: visible !important;
  }
}
</style>
""",
        unsafe_allow_html=True,
    )
