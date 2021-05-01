"""
This is an interactive plot for the Gamma Distribution
"""
import streamlit as st
import tensorflow_probability as tfp
import tensorflow as tf
import plotly.graph_objects as go

def main():
    st.set_page_config(
        page_title="Interactive Gamma Plot",
        page_icon="https://user-images.githubusercontent.com/27728103/116659788-6a611a00-a992-11eb-9cc7-ce02db99f106.png",
        layout="wide",
    )

    col_left, col_right = st.beta_columns((1, 1))

    col_left.title("Interactive Gamma Plot")

    col_left.latex("""
        \\text{{Gamma}}(
            \\tau
            ;
            \\alpha
            ,
            \\beta
        )
        =
        \\frac{{
            \\beta^{{\\alpha}}
        }} {{
            \\Gamma(\\alpha)
        }}
        \\tau^{{\\alpha - 1}}
        e^{{-\\beta \\tau}}
    """)

    # Alpha and beta have to be strictly greater than 0
    alpha = col_left.slider(
        label="alpha value",
        min_value=0.1,
        max_value=5.0,
        value=2.0,
        step=0.1,
    )
    beta = col_left.slider(
        label="beta value",
        min_value=0.1,
        max_value=5.0,
        value=2.0,
        step=0.1,
    )

    # Display the Gamma Function with plugged in values
    alpha_minus_1 = alpha - 1
    negative_beta = - beta
    col_left.latex(f"""
          p(\\tau)
            =
            \\text{{Gamma}}
            (
                \\tau
                ;
                \\alpha = {alpha:}
                ,
                \\beta = {beta:}
            )
            \\\\
            =
            \\frac {{
                {beta:}^{{{alpha:}}}
            }} {{
                \\Gamma({alpha:})
            }}
            \\tau^{{{alpha_minus_1:}}}
            e^{{{negative_beta:} \\tau}}
    """)

    # Watermark
    col_left.markdown("---")
    col_left.markdown("""
        Made with ❤️ by [Machine Learning & Simulation](https://www.youtube.com/channel/UCh0P7KwJhuQ4vrzc3IRuw4Q)
    """)

    # Start at small number to cover full curve
    x = tf.linspace(0.00000000000001, 10, 300)
    x = tf.cast(x, tf.float32)
    y = tfp.distributions.Gamma(alpha, beta).prob(x)
    figure = go.Figure(
        data=go.Scatter(x=x, y=y, mode="lines", line_color="orange", line_width=5),
        layout=go.Layout(yaxis_range=[-0.1, 1.0], xaxis_range=[-0.5, 10.5], height=600),
    )
    col_right.plotly_chart(figure, use_container_width=True)

if __name__ == "__main__":
    main()