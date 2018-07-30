"""
Provides various utils.

Cleans data moving to Nan from 'Unk' and '-1'
"""

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

PERM_ITTERATIONS = 10000

def clean_df(df, fields, vrates):
    """
    Returns the voter rate columns with Nan not -1
    and fields in fields have missing values replaced with NaNs.
    """
    # converting cleaned not known data ie 'UNK' to NaNs
    for f in fields:
        # replacing 'UNK' with Nan to indicate no data if fields has any 'UNK'
        if 'UNK' in df.loc[:, f].value_counts().index.values.astype(str):
            df.loc[:, f].replace('UNK', np.NaN, inplace=True)

    # replacing '-1' indicating no data for vote with NaN
    for c in df[vrates]:
        df[c].replace(-1, np.NaN, inplace=True)

    return df


def show_vote_rate_and_summary(df, lab, title):
    """
    Draws the horizontal histograms for the data provided in df '_nVotesPct' in columns one per hist required.
    labels for each data column and a title are passed in and used.
    """
    vrs = df.columns
    n_plt = len(vrs)

    fig, axes = plt.subplots(n_plt, 1, sharex=True, figsize=(20, n_plt * 2))

    # Vote in every election, or 5/6 then you're in the top Always vote' category
    # Vote in 1/2, 2/4, 3/6, 3/5 or 2/5 then in the middle category
    # Vote in no election or 1/6 then in the bottom 'Never vote' category
    edges = [0, 0.19, 0.39, 0.61, 0.81, 1]

    n = [0 for i in range(n_plt)]
    for i, (e, l) in enumerate(zip(vrs, lab)):
        # drawing the graph
        _ = axes[i].hist(df[e].dropna(), density=True, bins=edges, orientation='horizontal',
                         label=l, alpha=0.7)
        # calculating the bucket counts
        n[i], _ = np.histogram(df[e].dropna(), bins=edges)
    ap = {'arrowstyle': '->', 'color': 'gray'}
    for i, ax in enumerate(axes):
        ax.legend(loc='center right')
        ax.annotate('Voters who always Vote', xy=[1, 0.8], xytext=[1.1, 0.61], arrowprops=ap, color='gray')
        ax.annotate('Voters who never Vote', xy=[1, 0.2], xytext=[1.1, 0.29], arrowprops=ap, color='gray')

    ylab = 'Vote Rate in last 6 elections'
    axes[n_plt - 1].set_xlabel('Probability Density Function (PDF)')
    _ = axes[0].set_title(title)
    _ = axes[0].set_ylabel(ylab)
    _ = axes[0].yaxis.set_label_coords(-0.03, -0.05 * n_plt)
    plt.show()

    # calculating and displaying the summary table
    order = ['Never', 'Under Half', 'Half', 'Over Half', 'Always']
    dw = pd.DataFrame(n, columns=order, index=lab).transpose()
    dw2 = pd.DataFrame()
    for e in lab:
        dw2[e + '_pct'] = round(dw[e] / sum(dw[e]), 3) * 100
    dw = pd.concat([dw, dw2], keys=('Number of Voters', 'Voters as a %'), axis=1)
    dw = dw.loc[order[::-1], :]
    dw.loc['Totals'] = dw.sum(axis=0)
    dw['Number of Voters'] = dw['Number of Voters'].astype('int')
    display(dw)
    return dw


def get_two_sample_ns(d0,d1):
    """
    Calculate the number of elections and successes as well as the percent success rate
    for the two provided dataframes. '_nVotesPos' and '_nVotes' columns are expected.
    """
    nVotesPos = [c for c in d0.columns if 'nVotesPos' in c][0]
    nVotes = nVotesPos[:-3]
    n0, s0 = d0[nVotesPos].sum(), d0[nVotes].sum()
    n1, s1 = d1[nVotesPos].sum(), d1[nVotes].sum()
    return s0,n0,(100*s0/n0),s1,n1,(100*s1/n1)


def diff_frac_votes(d0, d1):
    """Compute the difference in fraction of votes.
    Input two data frames with the nVotesPos and nVotes fields for election(s) of interest"""
    [s0,n0,_,s1,n1,_] = get_two_sample_ns(d0, d1)
    #print('diff frac votes:{}'.format((s0 / n0) - (s1 / n1)))
    return (s0 / n0) - (s1 / n1)


def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""
    data = pd.concat((data1, data2))
    permuted_data = data.sample(frac=1)
    return permuted_data[:len(data1)], permuted_data[len(data1):]


def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""
    perm_replicates = np.empty(size)
    for i in range(size):
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)
        # Compute the test statistic
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)
    return perm_replicates


def two_sample_perm_test_diff_frac_votes(d0, d1, ax, lbl, tail=1):
    """Complete a two sample permutation test using difference in fraction of votes as test statistic"""

    # Compute the test statistic from experimental data: empirical_diff_means
    empirical_diff_frac_votes = diff_frac_votes(d0, d1)

    # Draw 10,000 permutation replicates: perm_replicates
    perm_replicates = draw_perm_reps(d0, d1, diff_frac_votes, size=PERM_ITTERATIONS)

    # Plot test statistic histogram
    _ = ax.hist(perm_replicates, density=True, bins=1000)
    _ = ax.axvline(empirical_diff_frac_votes, color='red')
    _ = ax.set_xlabel('Difference in vote rate for category')
    _ = ax.set_ylabel('PDF')
    _ = ax.set_title('For {}'.format(lbl))

    # Compute p-value: p
    p = np.sum(perm_replicates >= empirical_diff_frac_votes) / len(perm_replicates)
    if p > 0.5:
        p = 1 - p

    lmt = 0.01 if tail == 1 else 0.005
    [nn_ll, nn_ul] = np.percentile(perm_replicates, [lmt * 100, (1 - lmt) * 100])

    _ = ax.axvline(nn_ll, color='orange', ls='--')
    _ = ax.axvline(nn_ul, color='orange', ls='--')

    ar_ht = ax.get_ylim()[1] * 0.9
    tx_x = -np.abs(nn_ul) + 0.5 * np.std(perm_replicates)
    text = '{} tail 99% Conf ({:.3f} : {:.3f})'.format(tail, nn_ll, nn_ul)
    ap = {'arrowstyle':'->', 'color':'gray'}
    ax.annotate(text, xy=[nn_ul, ar_ht * 0.7], xytext=[tx_x, ar_ht], color='w', alpha=0, arrowprops=ap)
    ax.annotate(text, xy=[nn_ll, ar_ht * 0.7], xytext=[tx_x, ar_ht], color='gray', arrowprops=ap)

    return p, empirical_diff_frac_votes, ax


def plot_hist_vote_rate_vs_field(ax, df, voteRatef, field, bins):
    """
    Spliting the provided data into sometimes, always and never voters and drawing
    a histogram of the three vote categories on the provided axis.
    """
    a, b = 0.3, bins
    df1 = pd.DataFrame(df[[voteRatef, field]]).rename(columns={voteRatef: 'VR'})
    always, sometimes, never, allv = (df1.VR == 1), (df1.VR < 1) & (df1.VR > 0), (df1.VR == 0), (df1.VR.notnull())

    rs, _, _ = ax.hist(df1.loc[sometimes, field], bins=b, alpha=a, label='SomeTimesVoters')
    ra, _, _ = ax.hist(df1.loc[always, field], bins=b, alpha=a, label='AlwaysVoters')
    rn, _, _ = ax.hist(df1.loc[never, field], bins=b, alpha=a, label='NeverVoters')
    rt, _, _ = ax.hist(df1.loc[allv, field], bins=b, alpha=a, label='Total #', histtype='step')

    ax.legend(loc='upper left')
    return ax, [ra, rs, rn, rt]