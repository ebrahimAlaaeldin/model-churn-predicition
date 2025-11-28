import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_gender_churn_distribution(df, gender_col="gender", churn_col="Churn"):
    """
    Creates two donut pie charts side-by-side showing:
      - Gender distribution
      - Churn distribution

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    gender_col : str
        Column name for gender.
    churn_col : str
        Column name for churn.

    Returns
    -------
    None
    """
    
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{'type':'domain'}, {'type':'domain'}]]
    )

    # Gender pie
    fig.add_trace(
        go.Pie(
            labels=df[gender_col].value_counts().index,
            values=df[gender_col].value_counts().values,
            name="Gender"
        ),
        1, 1
    )

    # Churn pie
    fig.add_trace(
        go.Pie(
            labels=df[churn_col].value_counts().index,
            values=df[churn_col].value_counts().values,
            name="Churn"
        ),
        1, 2
    )

    # Formatting
    fig.update_traces(
        hole=.4,
        hoverinfo="label+percent+name",
        textfont_size=16
    )

    fig.update_layout(
        title_text="Gender and Churn Distributions",
        annotations=[
            dict(text='Gender', x=0.19, y=0.5, font_size=20, showarrow=False),
            dict(text='Churn', x=0.82, y=0.5, font_size=20, showarrow=False)
        ]
    )

    fig.show()


import plotly.express as px

def plot_churn_vs_contract(df, churn_col="Churn", contract_col="Contract",
                           churn_map=None, contract_map=None,
                           title="Churn Distribution w.r.t. Contract Type"):
    """
    Plots a grouped histogram showing Churn distribution by Contract type.

    Parameters
    ----------
    df : pandas.DataFrame
        The processed DataFrame.
    churn_col : str
        Name of the churn column.
    contract_col : str
        Name of the contract column.
    churn_map : dict or None
        Optional mapping for churn values (e.g., {0:'No',1:'Yes'}).
    contract_map : dict or None
        Optional mapping for contract categories.
    title : str
        Title of the plot.

    Returns
    -------
    None
    """

    df_plot = df.copy()

    # Apply label mappings if provided
    if churn_map is not None:
        df_plot[churn_col] = df_plot[churn_col].map(churn_map)

    if contract_map is not None:
        df_plot[contract_col] = df_plot[contract_col].map(contract_map)

    # Create histogram
    fig = px.histogram(
        df_plot,
        x=churn_col,
        color=contract_col,
        barmode='group',
        labels={'x': 'Churn'}
    )

    # Update layout
    fig.update_layout(
        width=700,
        height=500,
        bargap=0.1,
        title_text=f"<b>{title}</b>"
    )

    fig.show()
    
    
    
import plotly.graph_objects as go

import plotly.graph_objects as go

def plot_payment_method_distribution(df, payment_col="PaymentMethod", payment_map=None):
    """
    Plots a donut pie chart showing the distribution of payment methods.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    payment_col : str
        Column name containing payment method values.
    payment_map : dict or None
        Optional mapping to convert encoded values into readable labels.

    Returns
    -------
    None
    """

    df_plot = df.copy()

    # Apply mapping if provided (e.g., {0:'Bank transfer', 1:'Credit card', ...})
    if payment_map is not None:
        df_plot[payment_col] = df_plot[payment_col].map(payment_map)

    # Count values
    counts = df_plot[payment_col].value_counts()
    labels = counts.index
    values = counts.values

    # Create pie chart
    fig = go.Figure(data=[
        go.Pie(labels=labels, values=values, hole=0.3)
    ])

    # Styling
    fig.update_layout(
        title_text="<b>Payment Method Distribution</b>",
        legend_title_text="Payment Method"
    )

    fig.show()


import plotly.express as px

import plotly.express as px

def plot_churn_vs_category(df, churn_col, category_col, 
                           churn_map=None, category_map=None, 
                           title=None):
    """
    Plots a grouped histogram showing churn distribution against any category.
    
    Mapping must be provided from outside to avoid unexpected NaN values.
    """

    df_plot = df.copy()

    # Apply churn mapping (if provided)
    if churn_map is not None:
        df_plot[churn_col] = df_plot[churn_col].map(churn_map)

    # Apply category mapping (if provided)
    if category_map is not None:
        df_plot[category_col] = df_plot[category_col].map(category_map)

    # Default title
    if title is None:
        title = f"<b>Churn Distribution w.r.t. {category_col}</b>"

    fig = px.histogram(
        df_plot,
        x=churn_col,
        color=category_col,
        barmode='group',
        labels={'x': 'Churn'}
    )

    fig.update_layout(width=700, height=500, bargap=0.1, title_text=title)
    fig.show()
