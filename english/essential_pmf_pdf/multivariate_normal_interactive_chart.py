"""
This is an interactive web plot for contours of the 2D Multivariate Normal.

The user can move around the mean and change the components in the covariance
matrix by the help of sliders.

A check of symmetric positive definiteness for the covariance matrix is performed.
"""

import streamlit as st
import plotly.graph_objects as go
import tensorflow as tf
import tensorflow_probability as tfp


def main():
    st.set_page_config(
        page_title="Multivariate Normal",
        layout="wide",
        page_icon="https://user-images.githubusercontent.com/27728103/116659788-6a611a00-a992-11eb-9cc7-ce02db99f106.png"
    )

    st.title("Interactive Plot for the Multivariate Normal")

    col_left, col_right = st.beta_columns((1, 1))
    col_left.latex("""
        f(x) = \\frac{{
            1
        }} {{
            \\sqrt{{
                (2 \\pi)^k
                \\det(\\mathbf{{\\Sigma}})
            }}
        }}
        \\exp(
            - \\frac{{1}}{{2}}
            (\\vec{{x}} - \\vec{{\\mu}})^T
            \\mathbf{{\\Sigma}}^{{-1}}
            (\\vec{{x}} - \\vec{{\\mu}})
        )
    """)

    # Define all the sliders
    # -> Mus can take any values
    # -> Variances have to be strictly positive
    # -> Covariances can take any value
    mean_x = col_left.slider(
        label="x position of mean",
        min_value=-3.0,
        max_value=3.0,
        value=2.0,
        step=0.1,
    )
    mean_y = col_left.slider(
        label="y position of mean",
        min_value=-3.0,
        max_value=3.0,
        value=1.0,
        step=0.1,
    )
    variance_x = col_left.slider(
        label="Variance in x",
        min_value=0.1,
        max_value=3.0,
        value=1.0,
        step=0.1,
    )
    variance_y = col_left.slider(
        label="Variance in y",
        min_value=0.1,
        max_value=3.0,
        value=1.0,
        step=0.1,
    )
    covariance_xy = col_left.slider(
        label="Covariance in xy",
        min_value=-3.0,
        max_value=3.0,
        value=0.0,
        step=0.1,
    )

    # Watermark
    col_left.markdown("""
        Made with ❤️ by [Machine Learning & Simulation](https://www.youtube.com/channel/UCh0P7KwJhuQ4vrzc3IRuw4Q)
    """)

    # Display mean and covariance matrix with check for spd
    col_right.latex(f"""
        \\vec{{\\mu}}
        =
        \\begin{{bmatrix}}
            {mean_x:} \\\\
            {mean_y:} \\\\
        \\end{{bmatrix}}
        \\quad
        \\quad
        \\mathbf{{\Sigma}}
        =
        \\begin{{bmatrix}}
            {variance_x:} & {covariance_xy} \\\\
            {covariance_xy} & {variance_y} \\\\
        \\end{{bmatrix}}
    """)

    mu = [mean_x, mean_y]
    cov = [
        [variance_x, covariance_xy],
        [covariance_xy, variance_y],
    ]

    symmetric_positive_definite = True

    try:
        tf.linalg.cholesky(cov)
    except tf.errors.InvalidArgumentError:
        col_right.error("Covariance Matrix is not symmetric positive definite! 😟")
        symmetric_positive_definite = False

    if symmetric_positive_definite:
        col_right.success("Covariance Matrix is symmetric positive definite! 😃")

        # Display contours
        N_POINTS_P_AXIS = 100
        x = tf.linspace(-5, 5, N_POINTS_P_AXIS)
        y = tf.linspace(-5, 5, N_POINTS_P_AXIS)
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        X, Y = tf.meshgrid(x, y)
        X = tf.reshape(X, -1)
        Y = tf.reshape(Y, -1)
        points_2d = tf.stack((X, Y), axis=1)
        prob_values = tfp.distributions.MultivariateNormalTriL(
            loc=mu,
            scale_tril=tf.linalg.cholesky(cov)
        ).prob(points_2d)
        Z = tf.reshape(prob_values, (N_POINTS_P_AXIS, N_POINTS_P_AXIS))

        figure = go.Figure(
            data=go.Contour(
                autocontour=True,
                x=x,
                y=y,
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
            ),
        )
        col_right.plotly_chart(figure, use_container_width=True)
        

if __name__ == "__main__":
    main()