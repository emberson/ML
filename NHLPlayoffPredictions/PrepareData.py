import numpy as np
import pandas as pd
import scipy.optimize as soptim
import sys
import os

# -------------------------------------------------------------------------------------------------------
# PARAMETERS
# -------------------------------------------------------------------------------------------------------

# Input directory containing raw game data 
input_root_dir = "./data/raw/"
input_template = "gamesYEAR_REPLACE.dat"

# Output directory to save prepared data
output_root_dir = "./data/prepared/"
output_template = "seriesYEAR_REPLACE.dat"

# Number of games to include in rolling statistics
Nrolling = 7

# Flag to use effective goals in power-play and penalty kill
USE_SPECIAL_TEAM_EFFECTIVE_GOALS = True

# Flag to switch between using raw 5v5 stats or score-venue-adjusted (SVA) stats
USE_SVA_STATS = True

# NHL seasons to prepare data 
seasons = np.arange(2008, 2018+1)

# -------------------------------------------------------------------------------------------------------
# FUNCTIONS
# -------------------------------------------------------------------------------------------------------

def ReadRawData(ddir, dfile, year, rep_string="YEAR_REPLACE"):
    """
    Read dataframe containing raw game data.
    """

    ff = ddir + dfile.replace(rep_string, str(year))
    df = pd.read_csv(ff, index_col=0, sep="\t")

    return df

def FillMissingValues(df):
    """
    Handle missing values by replacing NA's with mean values for each team over the season 
    type (i.e., if regular season then average over all regular season games and if a playoff
    series then average over the other games in that playoff series)
    """

    # Make a copy of the original dataframe (needed for finding averages)
    df0 = df.copy()

    # Determine games with missing values
    ginds = df[df.isnull().any(axis=1)].index.values.astype(int)

    #
    # Handle each game one at a time
    #

    for index in ginds:

        # Determine teams and season type
        season = df.loc[index].Season
        hteam  = df.loc[index].Home
        ateam  = df.loc[index].Away

        # Pull out games that will be used to average the home and away stats
        hinds = df0.loc[((df0.Home == hteam) | (df0.Away == hteam)) & (df0.Season == season)]
        ainds = df0.loc[((df0.Home == ateam) | (df0.Away == ateam)) & (df0.Season == season)]

        # Fill in missing data with relevant home or away mean value 
        stats = df.columns[df.loc[index].isna()].tolist()
        for stat in stats:
            stype = stat.split("_")[0][-1] ; assert stype == "F" or stype == "A"
            if stype == "F": df.loc[index,stat] = hinds[stat].mean()
            elif stype == "A": df.loc[index,stat] = ainds[stat].mean()

    return df

def BradleyTerryModel(df, teams): 
    """
    Runs Bradley Terry model to determine team strengths (stored in array beta) as well
    as the home-ice advantage bias (alpha) based on all season games.
    Code allows functionality for ties which are counted as half a win, though this will
    not actually occur for post-2000 NHL seasons where all games are decided in OT/SO.
    """

    nteams = teams.shape[0]

    # Return if there are no games to analyze
    if nteams == 0: return 0, 0

    # Get team indices for home and away teams
    hinds = np.searchsorted(teams, np.array(df.Home))
    ainds = np.searchsorted(teams, np.array(df.Away))

    # Y stores the game results (0 for home team loss; 1 for home team win; 0.5 for tie)
    dftemp = df.assign(Y=0.)
    dftemp.loc[(dftemp.GF  > dftemp.GA),"Y"] = 1.0
    dftemp.loc[(dftemp.GF == dftemp.GA),"Y"] = 0.5
    Y = np.array(dftemp.Y)

    # Maximize the Bradley-Terry log-likelihood function
    res = soptim.minimize(BTLogLikelihood, np.zeros(nteams+1), args=(Y, hinds, ainds), method="BFGS") 
    sol = res.x

    # Pull out alpha (home-ice bias) and team strengths 
    alpha = sol[-1]
    beta  = sol[:-1] 

    # Normalize beta so lowest scoring team has beta = 0
    beta -= beta.min()

    return alpha, beta

def BTLogLikelihood(beta, Y, hinds, ainds):
    """
    Returns (negative of) the Bradley-Terry log-likelihood. Last index of beta is alpha (home-ice bias)
    """

    alpha  = beta[-1]    
    params = alpha + beta[hinds] - beta[ainds]
    l = np.sum(Y*params - np.log(1.+np.exp(params)))

    # Apply negative so that scipy minimization will correspond to a maximization
    l *= -1

    return l

def ComputeSeasonTeamStats(df0, season=0):
    """
    Use game data to derive season statistics for each team. Defaults to looking only
    at regular season (Season == 0). If season < 0 then looks at all games.
    """

    global USE_SVA_STATS

    # Pull out only relevant season
    if season >= 0: df = df0[df0.Season == season]
    else: df = df0.copy()

    # Determine set of NHL teams 
    teams  = np.sort(np.unique(np.append(df.Home, df.Away)))
    nteams = teams.shape[0]

    # Return None if there are no games to analyze
    if nteams == 0: return None

    # Initialize dataframe with team names
    ds = pd.DataFrame.from_dict({"Team":teams})

    # Initialize numpy arrays containing season stats for each team
    points = np.zeros(nteams, dtype="float64")
    gf     = np.zeros(nteams, dtype="float64")
    ga     = np.zeros(nteams, dtype="float64")
    cf_5v5 = np.zeros(nteams, dtype="float64")
    ca_5v5 = np.zeros(nteams, dtype="float64")
    ff_5v5 = np.zeros(nteams, dtype="float64")
    fa_5v5 = np.zeros(nteams, dtype="float64")
    sf_5v5 = np.zeros(nteams, dtype="float64")
    sa_5v5 = np.zeros(nteams, dtype="float64")
    gf_5v5 = np.zeros(nteams, dtype="float64")
    ga_5v5 = np.zeros(nteams, dtype="float64")
    cf_sva = np.zeros(nteams, dtype="float64")
    ca_sva = np.zeros(nteams, dtype="float64")
    ff_sva = np.zeros(nteams, dtype="float64")
    fa_sva = np.zeros(nteams, dtype="float64")
    sf_sva = np.zeros(nteams, dtype="float64")
    sa_sva = np.zeros(nteams, dtype="float64")
    gf_sva = np.zeros(nteams, dtype="float64")
    ga_sva = np.zeros(nteams, dtype="float64")
    gf_pp  = np.zeros(nteams, dtype="float64")
    ga_pk  = np.zeros(nteams, dtype="float64")
    shp_pp = np.zeros(nteams, dtype="float64")
    svp_pk = np.zeros(nteams, dtype="float64")
    
    #
    # Compute statistics normalized to the number of games played for each team 
    #

    for i in range(nteams):

        team = teams[i]

        # Separate dataframe for home and away games 
        dh = df.loc[df.Home == team]
        da = df.loc[df.Away == team]

        # Determine points from game results
        points[i] = ComputePoints(dh, da)

        # Determine total goals for and against
        gf[i], ga[i] = CumulativeStatistics(dh, da, "GF", "GA")

        # Determine shot statistics for 5v5 situations
        cf_5v5[i], ca_5v5[i] = CumulativeStatistics(dh, da, "CF_5v5", "CA_5v5")
        ff_5v5[i], fa_5v5[i] = CumulativeStatistics(dh, da, "FF_5v5", "FA_5v5")
        sf_5v5[i], sa_5v5[i] = CumulativeStatistics(dh, da, "SF_5v5", "SA_5v5")
        gf_5v5[i], ga_5v5[i] = CumulativeStatistics(dh, da, "GF_5v5", "GA_5v5")

        # Determine shot statistics for 5v5 situations with score and venue adjusted
        cf_sva[i], ca_sva[i] = CumulativeStatistics(dh, da, "CF_SVA", "CA_SVA")
        ff_sva[i], fa_sva[i] = CumulativeStatistics(dh, da, "FF_SVA", "FA_SVA")
        sf_sva[i], sa_sva[i] = CumulativeStatistics(dh, da, "SF_SVA", "SA_SVA")
        gf_sva[i], ga_sva[i] = CumulativeStatistics(dh, da, "GF_SVA", "GA_SVA")

        # Compute goals for (agains) and shooting (save) percentage on power-play (penalty kill) 
        gf_pp[i], ga_pk[i], shp_pp[i], svp_pk[i] = ComputeSpecialTeamStats(dh, da)

    #
    # Compute some derived quantities
    #

    shp_5v5 = gf_5v5/sf_5v5
    svp_5v5 = 1.0 - ga_5v5/sa_5v5
    pdo_5v5 = shp_5v5 + svp_5v5
    shp_sva = gf_sva/sf_sva
    svp_sva = 1.0 - ga_sva/sa_sva
    pdo_sva = shp_sva + svp_sva
    pdo_stm = shp_pp + svp_pk

    #
    # Run Bradley-Terry model to determine team strengths over all games
    #

    alpha, beta = BradleyTerryModel(df, teams)

    #
    # Summarize results into a dataframe storing either raw 5v5 stats or the score-venue-adjusted ones
    #

    if USE_SVA_STATS:
        ds = ds.assign(Points=points, Beta=beta,
                   GF=gf_sva, GA=ga_sva, SHP=shp_sva, SVP=svp_sva, PDO=pdo_sva, \
                   CF=cf_sva, CA=ca_sva, FF=ff_sva, FA=fa_sva, \
                   GF_PP=gf_pp, GA_PK=ga_pk, SHP_PP=shp_pp, SVP_PK=svp_pk, PDO_STM=pdo_stm)
    else:
        ds = ds.assign(Points=points, Beta=beta,
                   GF=gf_5v5, GA=ga_5v5, SHP=shp_5v5, SVP=svp_5v5, PDO=pdo_5v5, \
                   CF=cf_5v5, CA=ca_5v5, FF=ff_5v5, FA=fa_5v5, \
                   GF_PP=gf_pp, GA_PK=ga_pk, SHP_PP=shp_pp, SVP_PK=svp_pk, PDO_STM=pdo_stm)

    return ds

def ComputePoints(dh, da):
    """
    Compute points from wins, ties, and overtime/shootout losses.
    dh and da are dataframes containing home and away games, respectively.
    Points are normalized to the number of games. 
    """

    # Winning games
    nhw = dh.loc[(dh.GF > dh.GA)].shape[0]
    naw = da.loc[(da.GA > da.GF)].shape[0]

    # Tied games (this actually will not occur post 2000)
    nht = dh.loc[(dh.GF == dh.GA)].shape[0]
    nat = da.loc[(da.GA == da.GF)].shape[0]

    # Non-regulation losses
    nhl = dh.loc[(dh.Decision != "REG") & (dh.GF < dh.GA)].shape[0]
    nal = da.loc[(da.Decision != "REG") & (da.GA < da.GF)].shape[0]

    # Sum the results
    points = 2*(nhw+naw) + 1*(nht+nat+nhl+nal)

    # Normalize to the number of games played
    nn = 1.0*(dh.shape[0] + da.shape[0])

    return points/nn

def CumulativeStatistics(dh, da, sh, sa):
    """
    Return the sum of stat over all home and away games.
    """ 

    nn = 1.0*(dh.shape[0] + da.shape[0])
    vf = dh[sh].sum() + da[sa].sum()
    va = dh[sa].sum() + da[sh].sum()

    return vf/nn, va/nn 

def ComputeSpecialTeamStats(dh, da):
    """
    Compute goals for (against) on the power-play (penalty) kill. If desired, return
    only the effective goals which subtracts shorthanded goals in each situation.
    Also compute shooting (save) percentage on the power-play (penalty kill) using
    either the goals or effective goals
    """

    global USE_SPECIAL_TEAM_EFFECTIVE_GOALS

    # Total goals scored for and against during power-play
    ngf_pp = dh.GF_PP.sum() + da.GA_PK.sum()
    nga_pp = dh.GA_PP.sum() + da.GF_PK.sum()
    if USE_SPECIAL_TEAM_EFFECTIVE_GOALS:
        ngd_pp = ngf_pp - nga_pp
    else:
        ngd_pp = ngf_pp

    # Total goals scored for and against during penalty-kill
    ngf_pk = dh.GF_PK.sum() + da.GA_PP.sum()
    nga_pk = dh.GA_PK.sum() + da.GF_PP.sum()
    if USE_SPECIAL_TEAM_EFFECTIVE_GOALS:
        ngd_pk = nga_pk - ngf_pk
    else:
        ngd_pk = nga_pk

    # Total shots for (against) on the power-play (penalty kill)
    nsf_pp = dh.SF_PP.sum() + da.SA_PK.sum()
    nsa_pk = dh.SA_PK.sum() + da.SF_PP.sum()

    # Effective shooting (save) percentage on power-play (penalty kill)
    # Set NaN if there were no powerplay or penalty kill situations
    if nsf_pp == 0:
        shp_pp = np.nan
    else:
        shp_pp = ngd_pp/(1.0*nsf_pp)
    if nsa_pk == 0:
        svp_pp = np.nan
    else:
        svp_pp = 1.0 - ngd_pk/(1.0*nsa_pk)

    # Normalize these to the total number of games played
    nn = 1.0*(dh.shape[0] + da.shape[0])

    return ngd_pp/nn, ngd_pk/nn, shp_pp, svp_pp

def GeneratePlayoffSeriesFeatures(dg, ds, N):
    """
    Return dataframe containing features for each playoff series using a combination
    of season stats, regular series match results, and rolling statistics.
    """

    # Determine total number of playoff series
    nseries = dg.Season.max()
    assert nseries == 15

    # Initialize vectors containing series metadata 
    team1  = np.zeros(nseries, dtype="object")
    team2  = np.zeros(nseries, dtype="object")
    home   = np.zeros(nseries, dtype="int32")
    result = np.zeros(nseries, dtype="int32")
    pround = np.zeros(nseries, dtype="int32")

    # Initialize vectors containing global regular season comparative stats
    points   = np.zeros(nseries, dtype="float64")
    betas    = np.zeros(nseries, dtype="float64")
    goalsf   = np.zeros(nseries, dtype="float64")
    goalsa   = np.zeros(nseries, dtype="float64")
    pdos     = np.zeros(nseries, dtype="float64")
    corsif   = np.zeros(nseries, dtype="float64")
    corsia   = np.zeros(nseries, dtype="float64")
    fenwickf = np.zeros(nseries, dtype="float64")
    fenwicka = np.zeros(nseries, dtype="float64")
    goalsfs  = np.zeros(nseries, dtype="float64")
    goalsas  = np.zeros(nseries, dtype="float64")
    pdoss    = np.zeros(nseries, dtype="float64")

    # Initialize vectors containing regular season matchup stats
    points_mt   = np.zeros(nseries, dtype="float64")
    goalsf_mt   = np.zeros(nseries, dtype="float64")
    goalsa_mt   = np.zeros(nseries, dtype="float64")
    pdos_mt     = np.zeros(nseries, dtype="float64")
    corsif_mt   = np.zeros(nseries, dtype="float64")
    corsia_mt   = np.zeros(nseries, dtype="float64")
    fenwickf_mt = np.zeros(nseries, dtype="float64")
    fenwicka_mt = np.zeros(nseries, dtype="float64")
    goalsfs_mt  = np.zeros(nseries, dtype="float64")
    goalsas_mt  = np.zeros(nseries, dtype="float64")
    pdoss_mt    = np.zeros(nseries, dtype="float64")
    ngames_mt   = np.zeros(nseries, dtype="int32")

    # Initialize vectors containing comparative stats for last N games
    points_rl   = np.zeros(nseries, dtype="float64")
    goalsf_rl   = np.zeros(nseries, dtype="float64")
    goalsa_rl   = np.zeros(nseries, dtype="float64")
    pdos_rl     = np.zeros(nseries, dtype="float64")
    corsif_rl   = np.zeros(nseries, dtype="float64")
    corsia_rl   = np.zeros(nseries, dtype="float64")
    fenwickf_rl = np.zeros(nseries, dtype="float64")
    fenwicka_rl = np.zeros(nseries, dtype="float64")
    goalsfs_rl  = np.zeros(nseries, dtype="float64")
    goalsas_rl  = np.zeros(nseries, dtype="float64")
    pdoss_rl    = np.zeros(nseries, dtype="float64")
    ngames_rl   = np.zeros(nseries, dtype="int32")

    #
    # Gather data for each playoff series
    #

    for i in range(nseries):

        # Pull out playoff series
        dpi = dg[dg.Season == i+1]

        # Gather metadata for this playoff series
        team1i, team2i, homei, resulti, datei = PlayoffSeriesMetaData(dpi)
        team1[i]  = team1i
        team2[i]  = team2i
        home[i]   = homei
        result[i] = resulti
        pround[i] = GetPlayoffRound(i+1)

        # Get global regular season statistics for each team 
        pt1, bt1, gf1, ga1, pdo1, cf1, ca1, ff1, fa1, gfs1, gas1, pdos1 = CollectTeamStats(ds, team1i)
        pt2, bt2, gf2, ga2, pdo2, cf2, ca2, ff2, fa2, gfs2, gas2, pdos2 = CollectTeamStats(ds, team2i)
        points[i]   = ComparePoints(pt1, pt2)
        betas[i]    = CompareBetas(bt1, bt2)
        goalsf[i]   = CompareGoals(gf1, gf2)
        goalsa[i]   = CompareGoals(ga1, ga2)
        pdos[i]     = ComparePDOs(pdo1, pdo2)
        corsif[i]   = CompareShots(cf1, cf2)
        corsia[i]   = CompareShots(ca1, ca2)
        fenwickf[i] = CompareShots(ff1, ff2)
        fenwicka[i] = CompareShots(fa1, fa2)
        goalsfs[i]  = CompareGoals(gfs1, gfs2)
        goalsas[i]  = CompareGoals(gas1, gas2)
        pdoss[i]    = ComparePDOs(pdos1, pdos2)

        # Get regular season mathcup results for the two teams
        dm = GetSeasonMatchups(dg, team1i, team2i)
        ngames_mt[i] = dm.shape[0]
        if dm.shape[0] > 0: 
            dm = ComputeSeasonTeamStats(dm)
            pt1, bt1, gf1, ga1, pdo1, cf1, ca1, ff1, fa1, gfs1, gas1, pdos1 = CollectTeamStats(dm, team1i)
            pt2, bt2, gf2, ga2, pdo2, cf2, ca2, ff2, fa2, gfs2, gas2, pdos2 = CollectTeamStats(dm, team2i)
            points_mt[i]   = ComparePoints(pt1, pt2)
            goalsf_mt[i]   = CompareGoals(gf1, gf2)
            goalsa_mt[i]   = CompareGoals(ga1, ga2)
            pdos_mt[i]     = ComparePDOs(pdo1, pdo2)
            corsif_mt[i]   = CompareShots(cf1, cf2)
            corsia_mt[i]   = CompareShots(ca1, ca2)
            fenwickf_mt[i] = CompareShots(ff1, ff2)
            fenwicka_mt[i] = CompareShots(fa1, fa2)
            goalsfs_mt[i]  = CompareGoals(gfs1, gfs2)
            goalsas_mt[i]  = CompareGoals(gas1, gas2)
            pdoss_mt[i]    = ComparePDOs(pdos1, pdos2)
        else:
            points_mt[i]   = np.nan
            goalsf_mt[i]   = np.nan
            goalsa_mt[i]   = np.nan
            pdos_mt[i]     = np.nan
            corsif_mt[i]   = np.nan
            corsia_mt[i]   = np.nan
            fenwickf_mt[i] = np.nan
            fenwicka_mt[i] = np.nan
            goalsfs_mt[i]  = np.nan
            goalsas_mt[i]  = np.nan
            pdoss_mt[i]    = np.nan

        # Get rolling statistics from the previous N games (can include playoff games)
        dn1 = GatherLastNGames(dg, team1i, datei, N)
        dn2 = GatherLastNGames(dg, team2i, datei, N)
        dn1 = ComputeSeasonTeamStats(dn1, season=-1)
        dn2 = ComputeSeasonTeamStats(dn2, season=-1)
        pt1, bt1, gf1, ga1, pdo1, cf1, ca1, ff1, fa1, gfs1, gas1, pdos1 = CollectTeamStats(dn1, team1i)
        pt2, bt2, gf2, ga2, pdo2, cf2, ca2, ff2, fa2, gfs2, gas2, pdos2 = CollectTeamStats(dn2, team2i)
        points_rl[i]   = ComparePoints(pt1, pt2)
        goalsf_rl[i]   = CompareGoals(gf1, gf2)
        goalsa_rl[i]   = CompareGoals(ga1, ga2)
        pdos_rl[i]     = ComparePDOs(pdo1, pdo2)
        corsif_rl[i]   = CompareShots(cf1, cf2)
        corsia_rl[i]   = CompareShots(ca1, ca2)
        fenwickf_rl[i] = CompareShots(ff1, ff2)
        fenwicka_rl[i] = CompareShots(fa1, fa2)
        goalsfs_rl[i]  = CompareGoals(gfs1, gfs2)
        goalsas_rl[i]  = CompareGoals(gas1, gas2)
        pdoss_rl[i]    = ComparePDOs(pdos1, pdos2)
        ngames_rl[i]   = N

    #
    # Store the results in a dataframe
    #

    dp = pd.DataFrame.from_dict({"Team1":team1})
    dp = dp.assign(Team2=team2, Round=pround, Home=home, Result=result, 
                   PP=points, BB=betas, GF=goalsf, GA=goalsa, PDO=pdos, CF=corsif, CA=corsia, \
                   FF=fenwickf, FA=fenwicka, GFST=goalsfs, GAST=goalsas, PDOST=pdoss, \
                   Matches=ngames_mt, PP_M=points_mt, GF_M=goalsf_mt, GA_M=goalsa_mt, PDO_M=pdos_mt, CF_M=corsif_mt, CA_M=corsia_mt, \
                   FF_M=fenwickf_mt, FA_M=fenwicka_mt, GFST_M=goalsfs_mt, GAST_M=goalsas_mt, PDOST_M=pdoss_mt, \
                   Nlast=ngames_rl, PP_N=points_rl, GF_N=goalsf_rl, GA_N=goalsa_rl, PDO_N=pdos_rl, CF_N=corsif_rl, CA_N=corsia_rl, \
                   FF_N=fenwickf_rl, FA_N=fenwicka_rl, GFST_N=goalsfs_rl, GAST_N=goalsas_rl, PDOST_N=pdoss_rl)

    return dp

def PlayoffSeriesMetaData(dp):
    """
    Input is dataframe containing all games from a playoff series. Return teams involved
    in alphabetical order, whether the first team has home ice advantage, whether the first
    team won the playoff series, and the date of the first game of the series.
    """

    # Use first game to determine teams and whether the first team has home ice
    game1 = dp.iloc[0]
    team1, team2 = np.sort(np.array([game1.Home, game1.Away]))
    if team1 == game1.Home: home = True
    else: home = False
    date1 = game1.Date

    # Determine if team1 won the series
    wins1 = 0
    for i in range(dp.shape[0]):
        game = dp.iloc[i]
        if game.Home == team1:
            if game.GF > game.GA: wins1 += 1
        elif game.Away == team1:
            if game.GA > game.GF: wins1 += 1
        else: exit()
    if wins1 == 4: won = True
    else: won = False

    return team1, team2, home, won, date1

def GetPlayoffRound(n):
    """
    Determine which round of the playoffs this game is from.
    """

    if n <= 8:
        r = 1
    elif n <= 12:
        r = 2
    elif n <= 14:
        r = 3
    else:
        r = 4

    return r

def CollectTeamStats(df, team):
    """
    Return various statistics corresponding to the given team in df.
    """

    # Get index of team in df
    
    agh = df[(df.Team == team)].index.values.astype(int)
    if agh.shape[0] == 0:
        print("team: ", team)
        print(df)
        exit()

    index = df[(df.Team == team)].index.values.astype(int)[0]

    # Now collect various stats
    pt   = df.loc[index].Points
    bt   = df.loc[index].Beta
    gf   = df.loc[index].GF
    ga   = df.loc[index].GA
    pdo  = df.loc[index].PDO
    cf   = df.loc[index].CF
    ca   = df.loc[index].CA
    ff   = df.loc[index].FF
    fa   = df.loc[index].FA
    gfs  = df.loc[index].GF_PP
    gas  = df.loc[index].GA_PK
    pdos = df.loc[index].PDO_STM

    return pt, bt, gf, ga, pdo, cf, ca, ff, fa, gfs, gas, pdos

def Ratio(x, y):
    """
    Returns the ratio x/y
    """

    return (x/(1.*y))

def Fraction(x, y):
    """
    Returns the fraction of x.
    """

    return (x/(1.*(x+y)))

def Difference(x, y):
    """
    Returns the difference between x and y
    """

    return x-y

def ComparePoints(x, y):
    """
    Return feature that compares points between two teams.
    """

#    return Ratio(x, y) 
#    return Difference(x, y)
    return Fraction(x, y)

def CompareBetas(x, y):
    """
    Return feature that compares Bradley-Terry strengths between two teams.
    Have chosen to make this the probability that x beats y with the home-ice
    bias ignored.
    """

    pij = np.exp(x-y)/(1.0+np.exp(x-y))

    return pij

def CompareGoals(x, y):
    """
    Return feature that compares goals between two teams.
    """

#    return Ratio(x, y)
    return Difference(x, y)

def ComparePDOs(x, y):
    """
    Return feature that compares PDOs between two teams.
    """

    return Difference(x, y)

def CompareShots(x, y):
    """
    Return feature that compares shots between two teams.
    """

#    return Ratio(x, y)
    return Difference(x, y)

def GetSeasonMatchups(dg, t1, t2, S=0):
    """
    Return all games in a particular season (defaults to regular season) between the two teams
    """

    return dg[(((dg.Home == t1) & (dg.Away == t2)) | (dg.Home == t2) & (dg.Away == t1)) & (dg.Season == S)]

def GatherLastNGames(dg0, team, date, N):
    """
    Return the last N games played by the given team prior to the given date
    """

    return dg0[((dg0.Home == team) | (dg0.Away == team)) & (dg0.Date < date)][-N:]

def PrintInfo(dp):
    """
    Print a summary of the playoff series to screen.
    """

    print(dp[["Team1", "Team2", "Round", "Home", "Result", "Matches"]])

    rnan = dp[dp.isnull().any(axis=1)].index.values.astype(int)
    if rnan.shape[0] > 0:
        print("WARNING: Some playoff series have missing data,", rnan)


def SaveData(df, ddir, dtemp, year):
    """
    Save dataframe to CSV file.
    """

    ofile = ddir + dtemp.replace("YEAR_REPLACE", str(year))
    df.to_csv(ofile, na_rep="NA", sep="\t")
    print("Data saved to " + ofile)

# -------------------------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------------------------

#
# Create output directory if it does not yet exist
#

if not os.path.isdir(output_root_dir): os.makedirs(output_root_dir)

#
# Prepare data from raw data one season at a time 
#

for season in seasons:

    print("PREPARING DATA FOR " + str(season-1) + "-" + str(season) + " PLAYOFF SEASON")

    # Read dataframe containing raw game data 
    dg = ReadRawData(input_root_dir, input_template, season)

    # Handle any missing data
    dg = FillMissingValues(dg) 

    # Create dataframe containing season stats for each team 
    ds = ComputeSeasonTeamStats(dg)

    # Create dataframe containing stats for each playoff series 
    dp = GeneratePlayoffSeriesFeatures(dg, ds, Nrolling)

    # Print summary to screen
    PrintInfo(dp)

    # Save the result to CSV file
    SaveData(dp, output_root_dir, output_template, season) 

    print()

