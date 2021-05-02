"""
This is an interactive web plot for the contours of the Normal-Gamma Distribution.

We use this distribution as the conjugate prior for a unknown mean precision of
a Gaussian distribution.
"""

import streamlit as st
import plotly.graph_objects as go
import tensorflow as tf
import tensorflow_probability as tfp

def gauss_gamma(alpha_0, beta_0, mu_0, tau_0):
    """
    A generative model for the joint distribution of a Gauss-Gamma.
    """
    tau = yield tfp.distributions.JointDistributionCoroutine.Root(
        tfp.distributions.Gamma(alpha_0, beta_0)
    )
    mu = yield tfp.distributions.Normal(loc=mu_0, scale=tf.math.sqrt(1/(tau * tau_0)))

def main():
    st.set_page_config(
        page_title="Normal-Gamma Distribution Plot",
        layout="wide",
        page_icon="https://user-images.githubusercontent.com/27728103/116659788-6a611a00-a992-11eb-9cc7-ce02db99f106.png"
    )

    st.title("Interactive Plot for the Normal Gamma Distribution")

    col_left, col_right = st.beta_columns((1, 1))
    col_left.latex("""
        f(x) = \\frac{{
            \\beta_0^{{\\alpha_0}}
        }} {{
            \\Gamma(\\alpha_0)
        }}
        \\tau^{{\\alpha_0 - 1}}
        e^{{- \\beta_0 \\tau}}
        \\sqrt{{
            \\frac{{
                \\tau_0 \\tau
            }} {{
                2 \\pi
            }}
        }}
        \\exp(
            -
            \\frac{{
                \\tau_0
                \\tau
            }} {{
                2
            }}
            (
                \\mu
                -
                \\mu_0
            )^2
        )
    """)

    # Define all the sliders
    # -> alpha_0, beta_0 and tau_0 have to be striclty positive
    # -> mu_0 can take any value
    alpha_0 = col_left.slider(
        label="alpha_0 value",
        min_value=0.1,
        max_value=3.0,
        value=2.0,
        step=0.1,
    )
    beta_0 = col_left.slider(
        label="beta_0 value",
        min_value=0.1,
        max_value=3.0,
        value=1.0,
        step=0.1,
    )
    mu_0 = col_left.slider(
        label="mu_0 value",
        min_value=-3.0,
        max_value=3.0,
        value=1.0,
        step=0.1,
    )
    tau_0 = col_left.slider(
        label="tau_0 value",
        min_value=0.1,
        max_value=3.0,
        value=1.0,
        step=0.1,
    )

    # Watermark
    col_left.markdown("""
        Made with ❤️ by [Machine Learning & Simulation](https://www.youtube.com/channel/UCh0P7KwJhuQ4vrzc3IRuw4Q)
    """)


    # Display contours
    N_POINTS_P_AXIS = 100
    tau = tf.linspace(0.000001, 5.0, N_POINTS_P_AXIS)
    mu = tf.linspace(-5, 5, N_POINTS_P_AXIS)
    tau = tf.cast(tau, tf.float32)
    mu = tf.cast(mu, tf.float32)
    Tau, Mu = tf.meshgrid(tau, mu)
    Tau = tf.reshape(Tau, -1)
    Mu = tf.reshape(Mu, -1)
    points_2d = (Tau, Mu)
    prob_values = tfp.distributions.JointDistributionCoroutineAutoBatched(
        lambda : gauss_gamma(alpha_0, beta_0, mu_0, tau_0)
    ).prob(points_2d)
    Z = tf.reshape(prob_values, (N_POINTS_P_AXIS, N_POINTS_P_AXIS))

    figure = go.Figure(
        data=go.Contour(
            autocontour=True,
            x=tau,
            y=mu,
            z=Z,
            contours_coloring="heatmap",
            zmin=0.0,
            zmax=0.15,
            ncontours=10,
        ),
        layout=go.Layout(
            margin_l=0,
            margin_r=0,
            margin_b=0,
            margin_t=0,
            xaxis_title="tau",
            yaxis_title="mu",
        ),
    )
    col_right.plotly_chart(figure, use_container_width=True)
        

if __name__ == "__main__":
    main()