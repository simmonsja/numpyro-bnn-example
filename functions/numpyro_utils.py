import seaborn as sns
import matplotlib.pyplot as plt
import arviz as az
import jax.numpy as jnp

################################################################################
################################################################################
##############################   MAIN FUNCTIONS   ##############################
################################################################################
################################################################################

def calc_mean_hpdi(arviz_post, ci=0.89, y_scaler=None, mu_var='mu', sim_var='obs'):
    """
    Calculate the mean and highest posterior density interval (HPDI) for the 'mu' and 'obs' variables in the ArviZ posterior
    and posterior predictive objects.

    Parameters
    ----------
    arviz_post : ArviZ InferenceData object
        The posterior and posterior predictive samples.
    ci : float, optional
        The probability mass of the HPDI. Default is 0.89.
    y_scaler : Scaler object, optional
        A Scaler object to unscale the mean and HPDI values if the data was scaled before fitting the model. Default is None.
    mu_var : str, optional
        The name of the 'mu' variable in the ArviZ posterior object. Default is 'mu'.
    sim_var : str, optional
        The name of the 'obs' variable in the ArviZ posterior predictive object. Default is 'obs'.

    Returns
    -------
    mean_mu : ndarray
        The mean of the 'mu' variable.
    hpdi_mu : ndarray
        The HPDI of the 'mu' variable.
    hpdi_sim : ndarray
        The HPDI of the 'obs' variable.
    """
    # Define the dimensions that are common to both the posterior and posterior predictive objects
    base_dims = ['chain','draw']
    
    # Get the mean of the 'mu' variable in the posterior object
    mean_mu = arviz_post.posterior[mu_var].mean(dim=base_dims).values
    
    # Define the dimensions for the 'mu' and 'obs' variables that are not in the base dimensions
    mu_dims = [_ for _ in arviz_post.posterior[mu_var].coords.dims if not _ in base_dims]
    sim_dims = [_ for _ in arviz_post.posterior_predictive[sim_var].coords.dims if not _ in base_dims]
    
    # Calculate the HPDI for the 'mu' and 'obs' variables using the arviz.hdi() function
    hpdi_mu = az.hdi(
        arviz_post.posterior, hdi_prob=ci, var_names=[mu_var]
    ).transpose('hdi',*mu_dims)[mu_var].values
    hpdi_sim = az.hdi(
        arviz_post.posterior_predictive, hdi_prob=ci, var_names=[sim_var]
    ).transpose('hdi',*sim_dims)[sim_var].values

    # If a scaler object is provided, unscale the mean and HPDI values and reverse the log transform if necessary
    if not y_scaler is None:
        # Unscale the mean and HPDI values
        mean_mu = y_scaler.inverse_transform(mean_mu)
        hpdi_mu[0,...] = y_scaler.inverse_transform(hpdi_mu[0,...])
        hpdi_mu[1,...] = y_scaler.inverse_transform(hpdi_mu[1,...])
        hpdi_sim[0,...] = y_scaler.inverse_transform(hpdi_sim[0,...])
        hpdi_sim[1,...] = y_scaler.inverse_transform(hpdi_sim[1,...])
        # Reverse the log transform
        mean_mu = jnp.exp(mean_mu)
        hpdi_mu = jnp.exp(hpdi_mu)
        hpdi_sim = jnp.exp(hpdi_sim)
    
    # Return the mean and HPDI values for the 'mu' and 'obs' variables
    return mean_mu, hpdi_mu, hpdi_sim
        
################################################################################
################################################################################


def plot_prediction(df_Y, mean_mu, hpdi_mu, hpdi_sim, ci=0.89, save_loc=None):
    """
    Plot the modelled and observed y values with the modelled and simulated confidence intervals.

    Parameters
    ----------
    df_Y : pandas DataFrame
        The observed y values.
    mean_mu : ndarray
        The mean of the 'mu' variable.
    hpdi_mu : ndarray
        The HPDI of the 'mu' variable.
    hpdi_sim : ndarray
        The HPDI of the 'obs' variable.
    ci : float, optional
        The probability mass of the HPDI. Default is 0.89.
    save_loc : str, optional
        The file path to save the plot. Default is None.

    Returns
    -------
    None
    """
    # Set the seaborn style and context
    sns.set_style('ticks')
    sns.set_context('paper')
    
    # Create a new figure with the specified size
    fig = plt.figure(figsize=(7,3))
    # Create a new subplot for the plot
    ax1 = plt.subplot(111)
    # Plot the modelled y values
    ax1.plot(df_Y.index, mean_mu, label='Modelled')
    # Shade the area between the upper and lower bounds of the simulated confidence interval
    ax1.fill_between(df_Y.index, hpdi_sim[0,:], hpdi_sim[1,:], alpha=0.25, color='C1', label='Simulated {:.0f}% CI'.format(ci*100))
    # Shade the area between the upper and lower bounds of the modelled confidence interval
    ax1.fill_between(df_Y.index, hpdi_mu[0,:], hpdi_mu[1,:], alpha=0.5, color='C0', label='Modelled {:.0f}% CI'.format(ci*100))
    # Plot the observed y values
    ax1.plot(df_Y.index, df_Y.values, label='Observed')
    # Set the title, x-axis label, and y-axis label for the plot
    ax1.set_ylabel('Values')
    ax1.set_xlabel('Index')
    # Add a legend to the plot
    lgd = plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))
    # Save the plot if a file path is specified
    if save_loc is not None:
        plt.savefig(save_loc, bbox_extra_artists=(lgd,), bbox_inches='tight')
    # Show the plot
    plt.show()

################################################################################
################################################################################

def plot_prediction_scatter(df_Y, mean_mu, hpdi_mu, hpdi_sim, ci=0.89):
    """
    Plot the modelled and observed y values as a scatter plot with error bars.

    Parameters
    ----------
    df_Y : pandas DataFrame
        The observed y values.
    mean_mu : ndarray
        The mean of the 'mu' variable.
    hpdi_mu : ndarray
        The HPDI of the 'mu' variable.
    hpdi_sim : ndarray
        The HPDI of the 'obs' variable.
    ci : float, optional
        The probability mass of the HPDI. Default is 0.89.

    Returns
    -------
    None
    """
    # Set the seaborn style and context
    sns.set_style('ticks')
    sns.set_context('paper')

    # Create a new figure with the specified size
    fig = plt.figure(figsize=(4, 4))
    # Create a new subplot for the plot
    ax1 = plt.subplot(111)

    # Constrain x and y lims to be the same and equal to min max between modelled and observed
    ax1.set_xlim([min(df_Y.min(), mean_mu.min()), max(df_Y.max(), mean_mu.max())])
    ax1.set_ylim([min(df_Y.min(), mean_mu.min()), max(df_Y.max(), mean_mu.max())])
    # plot a 1:1 line red dashed
    ax1.plot([min(df_Y.min(), mean_mu.min()), max(df_Y.max(), mean_mu.max())], [min(df_Y.min(), mean_mu.min()), max(df_Y.max(), mean_mu.max())], '--', color='red')

    # Plot the modelled y values as a scatter plot
    ax1.plot(df_Y, mean_mu, 'o', label='Modelled')
    # Plot error bars for the modelled y values
    if not hpdi_sim is None: 
        ax1.errorbar(df_Y, mean_mu, yerr=[mean_mu - hpdi_sim[0,:], hpdi_sim[1,:] - mean_mu], fmt='none', alpha= 0.3, color='C0')

    # Set the title, x-axis label, and y-axis label for the plot
    ax1.set_ylabel('Modelled Values')
    ax1.set_xlabel('Observed Values')
    # Show the plot
    plt.show()

################################################################################
################################################################################



################################################################################
################################################################################


################################################################################
################################################################################