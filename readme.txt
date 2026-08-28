download pip install joblib
download pip install tensorflow
download pip install scikit-learn
download pip install streamlit-webrtc
download pip install av

Live camera connection:
- Local and deployed use default to Google's free public STUN server and do
  not require secrets.
- A TURN relay is optional for networks that cannot establish a direct WebRTC
  route. If one is needed, configure these root-level Streamlit secrets:
    TURN_URL = "turns:your-turn-host:443?transport=tcp"
    TURN_USERNAME = "your-username"
    TURN_CREDENTIAL = "your-password"
