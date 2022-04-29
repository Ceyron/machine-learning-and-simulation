import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import tensorflow_probability as tfp
import tensorflow as tf
from scipy import integrate

tfd = tfp.distributions

def joint_model(alpha, sigma):
    Z = yield tfd.JointDistributionCoroutine.Root(
        tfd.Exponential(
            rate=alpha,
            force_probs_to_zero_outside_support=True,
            name="latent_mu",
        )
    )
    X = yield tfd.Normal(
        loc=Z,
        scale=sigma,
    )

def main():
    st.set_page_config(
        page_title="Variational Inference Example",
        layout="wide",
        page_icon="https://user-images.githubusercontent.com/27728103/116659788-6a611a00-a992-11eb-9cc7-ce02db99f106.png"
    )

    col_left, col_right = st.columns((1, 1))
    col_left.title("VI Example")
    col_left.write("Using Exponential Normal Model")
    col_left.latex("""
        Z \\propto \\mathrm{{Exp}}(Z; \\alpha = 1)
        \\\\
        X \\propto \\mathcal{{N}}(X; \\mu = Z, \\sigma = 1)
    """)

    x_value_observed = col_left.slider(
        label="Observed X Value",
        min_value=-2.0,
        max_value=5.0,
        step=0.1,
        value=1.3,
    )

    alpha_set = 1.0
    sigma_set = 1.0

    joint_pdf = tfd.JointDistributionCoroutine(
        lambda : joint_model(alpha_set, sigma_set)
    )

    joint_pdf_fixed_to_data = lambda Z: joint_pdf.prob(Z, x_value_observed)

    Z_range = tf.linspace(-2.0, 5.0, 500)
    joint_pdf_prob_range = joint_pdf_fixed_to_data(Z_range)

    area_under_curve_approx = integrate.trapezoid(x=Z_range, y=joint_pdf_prob_range)

    col_left.latex(f"""
        \\int_{{Z=0}}^{{\\infty}}
        p(Z, X=D)
        \\mathrm{{d}} Z
        \\approx
        {area_under_curve_approx:1.3f}
    """)

    hypothetical_posterior_prob_range = joint_pdf_prob_range / area_under_curve_approx

    optimal_surrogate_posterior_parameter = (
        tf.sqrt(
            (
                (x_value_observed - alpha_set)**2
            ) / (
                4
            )
            +
            2
        )
        -
        (
            (x_value_observed - alpha_set)
        ) / 2
    )

    surrogate_posterior = tfp.distributions.Exponential(
        optimal_surrogate_posterior_parameter,
        force_probs_to_zero_outside_support=True,
        name="surrogate_posterior"
    )

    surrogate_posterior_prob_range = surrogate_posterior.prob(Z_range)

    # joint_fixed_to_data_figure = px.line(x=Z_range, y=prob_range)
    joint_fixed_to_data_figure = go.Figure([
        go.Scatter(x=Z_range, y=joint_pdf_prob_range, mode="lines", name="Joint"),
        go.Scatter(x=Z_range, y=hypothetical_posterior_prob_range, mode="lines", name="'True Posterior'"),
        go.Scatter(x=Z_range, y=surrogate_posterior_prob_range, mode="lines", name="Surrogate Posterior"),
    ])
    joint_fixed_to_data_figure.update_layout(
        xaxis_title="Z",
        yaxis_title="prob",
    )


    col_right.plotly_chart(joint_fixed_to_data_figure, use_container_width=True)
    col_right.markdown("*There is no true posterior for this model. Displayed is the joint numerically normalized.*")

    # Watermark
    col_left.write("")
    col_left.write("")
    col_left.markdown("""
        Made with ❤️ by [Machine Learning & Simulation](https://www.youtube.com/channel/UCh0P7KwJhuQ4vrzc3IRuw4Q)
    """)


if __name__ == "__main__":
    main()

