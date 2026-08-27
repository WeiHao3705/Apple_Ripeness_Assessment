import streamlit as st


def render_login() -> None:
    """Display the login screen.

    Authentication will be connected to Supabase in a later step.
    """
    st.title("🍎 Apple Ripeness System")
    st.caption("Sign in to scan apples and view your analysis history.")

    with st.container(border=True):
        st.subheader("Login")

        with st.form("login_form"):
            email = st.text_input(
                "Email address",
                placeholder="name@example.com",
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
            )
            submitted = st.form_submit_button(
                "Sign in",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            if not email or not password:
                st.warning("Please enter both your email address and password.")
            else:
                st.info("The login interface is ready. Supabase authentication is not connected yet.")

    st.caption("New user registration will be added with authentication.")
