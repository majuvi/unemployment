import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, beta

# Beta(a, b) parameters for specified mean and variance
beta_a = lambda mean, var: mean*(mean*(1-mean)/var-1)
beta_b = lambda mean, var: (1-mean)*(mean*(1-mean)/var-1)
Beta = lambda mean_p, var_p: beta(beta_a(mean_p, var_p), beta_b(mean_p, var_p))

# Generate a data set of unemployement sequences given entry and exit distributions
def sample_data(N, T, P, Q):
    data = []
    rates = []
    for sample in range(N):
        p, q = P.rvs(), Q.rvs()
        data.extend([(sample, time, spell, timein, unemployed, event) for (time, spell, timein, unemployed, event) in sample_sequence(T, p, q)])
        rates.append((sample, p, q))
    data = pd.DataFrame(data, columns=['sample', 'time', 'spell', 'timein', 'unemployed', 'event'])
    rates = pd.DataFrame(rates, columns=['sample', 'entry', 'exit'])
    return(data, rates)

# Generate a single sequence of observations given entry and exit rates
def sample_sequence(T, enter, exit):
    history = []
    spell = 0
    timein = 1
    enter = 1e-6 if enter < 1e-6 else enter
    exit = 1e-6 if exit < 1e-6 else exit
    steady_state = enter / (enter + exit)
    unemployed = bernoulli.rvs(steady_state)
    for time in range(T):
        event = bernoulli.rvs(exit if unemployed else enter)
        history.append((time, spell, timein, unemployed, event))
        timein += 1
        if event:
            spell += 1
            timein = 1
            unemployed = 1 - unemployed
    return history

# State occupancy statistics
def plot_data_states(data, color_0=(0,1,0), color_1=(1,0,0)):
    # Data set in long format
    data['time_next'] = data['time'] + 1
    N = len(data['sample'].unique())
    T = len(data['time'].unique())
    # 4 by 4 figure of marginal statistics
    fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, gridspec_kw = {'width_ratios': (4, 1), 'height_ratios': (4, 1)})
    # Sample timelines Sample x Time
    ax1.hlines(data['sample'], data['time'], data['time_next'], colors=[color_1 if unemployed else color_0 for unemployed in data['unemployed']])
    # State occupancy / Time
    state_freq = data.groupby('time')['unemployed'].agg('sum') / N
    ax2.vlines(state_freq.index, 0, state_freq.values, colors=color_1)
    ax2.vlines(state_freq.index, state_freq.values, 1, colors=color_0)
    # State occupancy / Sample
    durat_freq = data.groupby('sample')['unemployed'].agg('sum') / T
    ax3.hlines(durat_freq.index, 0, durat_freq.values, colors=color_1)
    ax3.hlines(durat_freq.index, durat_freq.values, 1, colors=color_0)
    ax4.axis('off')

# Plot entry and exit rates given distribution and samples
def plot_rates(P, Q, ps, qs, color_0=(0,1,0), color_1=(1,0,0)):
    x = np.linspace(0.0, 1.0, 1000)
    bins = np.linspace(0,1,101)
    plt.figure()
    plt.title('Entry and Exit distibutions')
    plt.plot(x, P.pdf(x), color=color_0, lw=2, alpha=1.0, label='unemployment entry probability')
    plt.hist(ps, bins=bins, color=color_0, alpha=0.4, label='', density=True)
    plt.plot(x, Q.pdf(x), color=color_1, lw=2, alpha=1.0, label='unemployment exit probability')
    plt.hist(qs, bins=bins, color=color_1, alpha=0.4, label='', density=True)
    plt.xlim(0.00, 0.25)
    plt.legend()

if __name__ == '__main__':
    # Entry and exit rates as Beta(mean, var) distributed
    mean_p, var_p = 0.0125, 0.02**2
    mean_q, var_q = 0.07, 0.02**2
    P = Beta(mean_p, var_p)
    Q = Beta(mean_q, var_q)

    # Generate a heterogeneous data set given samples (N), time (T)
    N, T = 100, 120
    data, rates = sample_data(N, T, P, Q)

    plot_rates(P, Q, rates['entry'], rates['exit'])
    plot_data_states(data)

    plt.show()

