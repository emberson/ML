import numpy as np
import pandas as pd
from urllib.request import urlopen
from bs4 import BeautifulSoup as bsoup
import sys
import os

# -------------------------------------------------------------------------------------------------------
# PARAMETERS
# -------------------------------------------------------------------------------------------------------

# Output data directory and file template 
data_root_dir = "./data/raw/"
data_template = "gamesYEAR_REPLACE.dat"

# NHL seasons to extract data frome 
seasons = np.arange(2008, 2018+1)

# Shot statistics to read from NaturalStatTrick for standard (5v5) play, powerplay, and penalty kill
# Definitions can be found at http://naturalstattrick.com/glossary.php?teams
shotstats = np.array(["CF", "CA", "FF", "FA", "SF", "SA", "SCF", "SCA", "HDCF", "HDCA", "GF", "GA"])

# -------------------------------------------------------------------------------------------------------
# FUNCTIONS 
# -------------------------------------------------------------------------------------------------------

def ProgressBar(i, N, blen=50):
    """
    Prints a progress bar to screen
    """

    tick = float(N) / blen

    s_len = int(np.round((i+1) / tick))
    s_per = int(np.round(100. * float(i+1) / N))
    s_pro = (s_len*"=").ljust(blen+1) + str(s_per) +"%\r"
    sys.stdout.write(s_pro)
    sys.stdout.flush()

    if (i+1) == N: print("")

def ParseHTML(url):
    """
    Use BeautifulSoup to parse html.
    """

    uclient = urlopen(url)
    page_html = uclient.read()
    uclient.close()
    soup = bsoup(page_html, "html.parser")

    return soup

def ReadHTMLTable(soup, tabid, url):
    """
    Find table with id in html and return as a pandas dataframe.
    """

    try:    
        table = soup.find("table", id=tabid)
        df    = pd.read_html(str(table))[0]
    except:
        print("ERROR: Could not find table with id", tabid, "on", url)
        exit()

    return df

def ExtractSeasonGames(year, rtableid="games", ptableid="games_playoffs", \
                             url_template="https://www.hockey-reference.com/leagues/NHL_YEAR_REPLACE_games.html"):
    """
    Parses the regular season and playoff results from www.hockey-reference.com
    Creates a single dataframe with final scores for both regular season and playoffs.
    """

    # Create the appropriate URL for the given season
    url = url_template.replace("YEAR_REPLACE", HockeyReferenceYearFormat(year))

    # Parse html code to read regular season and playoff results
    soup = ParseHTML(url)
    dfr = ReadHTMLTable(soup, rtableid, url)
    dfp = ReadHTMLTable(soup, ptableid, url)

    # Reformat dataframes
    dfr.drop(columns=["Att.","LOG","Notes"], inplace=True)
    dfp.drop(columns=["Att.","LOG","Notes"], inplace=True)
    dfr.rename({"Visitor":"Away", "G":"GA", "G.1":"GF"}, axis=1, inplace=True)
    dfp.rename({"Visitor":"Away", "G":"GA", "G.1":"GF"}, axis=1, inplace=True)
    dfr.rename({"Unnamed: 5": "Decision"}, axis=1, inplace=True)
    dfp.rename({"Unnamed: 5": "Decision"}, axis=1, inplace=True)
    dfr = dfr.assign(Season=0)
    dfp = dfp.assign(Season=1)
    dfr.fillna({"Decision": "REG"}, inplace=True)
    dfp.fillna({"Decision": "REG"}, inplace=True)
    assert len(dfr.Decision.unique()) <= 3 # Consistency check

    # Concatenate two dataframes to make single season data frame
    df = pd.concat([dfr,dfp], ignore_index=True)

    # Format team names in lower case with punctuation removed
    df.Away = df.Away.apply(TeamFMT)
    df.Home = df.Home.apply(TeamFMT)

    # Remove any instances where games were delayed to later dates (goals are null in this case)
    df = df.dropna(subset=["GA", "GF"])

    return df

def HockeyReferenceYearFormat(year):
    """
    Formats the NHL season year following the url convention of www.hockey-reference.com
    """

    return str(year)

def ExtractShotStatistics(df, stats, year, rind0=20001, sufst="_5v5", sufsa="_SVA", sufpp="_PP", sufpk="_PK", \
                          url_template="http://naturalstattrick.com/game.php?season=YEAR_REPLACE&game=GAME_REPLACE&view=limited"):
    """
    Uses www.naturalstattrick.com to add shot statistics to game dataframe. This is done after
    creating the www.hockey-reference.com dataframe since the former is easier to parse to determine
    the number of regular season and playoff games. This also serves as a useful consistency check. 
    """ 

    #
    # Add season into url_template
    #

    url_template = url_template.replace("YEAR_REPLACE", NaturalStatTrickYearFormat(year)) 

    #
    # Add columns for shot statistics to the dataframe
    #

    columns = { }
    for j in range(stats.shape[0]): columns[stats[j]+sufst] = 0
    for j in range(stats.shape[0]): columns[stats[j]+sufsa] = 0
    for j in range(stats.shape[0]): columns[stats[j]+sufpp] = 0
    for j in range(stats.shape[0]): columns[stats[j]+sufpk] = 0
    df = df.assign(**columns)

    #
    # Keep list of games where data was successfully extracted 
    #

    df = df.assign(found=0)

    #
    # Start with regular season games
    #

    # Count regular season games which are counted 0 in Season
    nreg = df.Season.value_counts()[0] 

    print("Accumulating statistics for " + str(nreg) + " regular season games") 

    for i in range(nreg):

        # Parse html of this page
        url = url_template.replace("GAME_REPLACE", str(rind0+i))
        soup = ParseHTML(url)

        # Read header to determine teams, date, and score
        date, ateam, hteam, agoal, hgoal = ParseNaturalStatTrick(soup.h1, soup.h2, url)

        # Determine the game this corresponds to in df 
        gindex = FindGameIndex(df, date, ateam, hteam, agoal, hgoal, url)

        if gindex >= 0:

            df, flagged = ParseNaturalStatTrickTables(df, soup, url, gindex, stats, sufst, sufsa, sufpp, sufpk)

            if flagged: print("WARNING: Some data was missing in url:", url) 
            else: df.loc[gindex,"found"] = 1

        ProgressBar(i, nreg)

    #
    # Now work through each of the playoff series 
    #

    # This should always be true
    nseries = 15
    npgames = df.Season.value_counts()[1]

    print("Accumulating statistics for " + str(nseries) + " playoff series containing " + str(npgames) + " games")

    np0 = 0 # Game ticker

    for i in range(nseries):

        # Determine url index for first game of this series
        sind0 = NaturalStatTrickPlayoffIndex(i)

        # Use first game in this series to mark Season column as the series index
        url = url_template.replace("GAME_REPLACE", str(sind0))
        soup = ParseHTML(url)
        date, ateam, hteam, agoal, hgoal = ParseNaturalStatTrick(soup.h1, soup.h2, url)
        sindex = FindSeriesIndices(df, ateam, hteam)
        df.loc[sindex,"Season"] = i+1

        # Now move through each game of the series at a time
        for k in range(sindex.shape[0]):
            url = url_template.replace("GAME_REPLACE", str(sind0+k))
            soup = ParseHTML(url)
            date, ateam, hteam, agoal, hgoal = ParseNaturalStatTrick(soup.h1, soup.h2, url)
            gindex = FindGameIndex(df, date, ateam, hteam, agoal, hgoal, url)
            if gindex >= 0:
                assert df.loc[gindex,"Season"] == i+1
                df, flagged = ParseNaturalStatTrickTables(df, soup, url, gindex, stats, sufst, sufsa, sufpp, sufpk)
                if flagged: print("WARNING: Some data was missing in url:", url)
                else: df.loc[gindex,"found"] = 1
            ProgressBar(np0, npgames) 
            np0 += 1

    # Consistency check that we got every game
    assert np0 == npgames

    #
    # Fill any missing games with null values
    #

    inds = df.loc[(df.found == 0)].index.values.astype(int)
    for i in range(inds.shape[0]):
        gindex = inds[i]
        for j in range(stats.shape[0]):
            stat = stats[j]
            statst = stat+sufst ; statsa = stat+sufsa ; statpp = stat+sufpp ; statpk = stat+sufpk
            df.loc[gindex,statst] = np.nan
            df.loc[gindex,statsa] = np.nan
            df.loc[gindex,statpp] = np.nan
            df.loc[gindex,statpk] = np.nan

    df.drop(columns=["found"], inplace=True)

    return df

def NaturalStatTrickPlayoffIndex(ns, pind0=30111, dpind0=10, dpsind0=100):
    """
    Determine the url index of the first game of the given playoff series index
    """

    if ns < 8: # First round
        ind = pind0 + ns*dpind0
    elif ns < 12: # Second round
        ind = pind0 + dpsind0 + (ns-8)*dpind0
    elif ns < 14: # Third round
        ind = pind0 + 2*dpsind0 + (ns-12)*dpind0
    else: # Fourth round
        ind = pind0 + 3*dpsind0 + (ns-14)*dpind0

    return ind

def NaturalStatTrickYearFormat(year):
    """
    Formats the NHL season year following the url convention of www.naturalstattrick.com
    """

    return str(year-1)+str(year)

def FindGameIndex(df, date, ateam, hteam, agoal, hgoal, url):
    """
    Return the index of the dataframe corresponding to the given game.
    Make sure this has exactly one instance in df. 
    """

    # Find index corresponding to given game details
    index = df.loc[(df.Date == date) & (df.Away == TeamFMT(ateam)) & (df.Home == TeamFMT(hteam))]
    if index.shape[0] != 1:
        # Try using date and goal information since some urls are missing team information
        index = df.loc[(df.Date == date) & (df.GA == agoal) & (df.GF == hgoal)]
        if index.shape[0] != 1:
            print("WARNING: Could not map game date:", date, "ateam:", ateam, "hteam:", hteam, "agoal:", agoal, "hgoal:", hgoal, "url:", url)
            return -1

    # Consistency check on the final score
    index = index.index.values.astype(int)[0]
    if df.loc[index,"GA"] != agoal or df.loc[index,"GF"] != hgoal:
        print("WARNING: Inconsistent final score GF:", df.loc[index,"GF"], hgoal, "GA:", df.loc[index,"GA"], agoal, "url: ", url)

    return index 

def FindSeriesIndices(df, team1, team2): 
    """
    Return indices of the dataframe corresponding to the playoff series between the two teams. 
    """

    indices = df.loc[(((df.Home == TeamFMT(team1)) & (df.Away == TeamFMT(team2))) | \
                     ((df.Home == TeamFMT(team2)) & (df.Away == TeamFMT(team1)))) & (df.Season == 1)]
    if indices.shape[0] < 4:
        print("ERROR: Could not determine playoff series indices: ", team1, team2)
        exit()

    return indices.index.values.astype(int)

def TeamFMT(team):
    """
    Removes any "." from team that causes an inconsitency between hockey-reference and naturalstattrick. Also cast to title. 
    """

    return str(team).replace(".","").title()

def ParseNaturalStatTrick(h1, h2, url):
    """
    Parse www.naturalstattrick.com to determine teams, score, and date.
    """

    try:
        # Determine home and away teams
        ateam, hteam = str(h1).split("@")
        ateam = ateam[4:].rstrip()
        hteam = hteam[:-5].lstrip()
        # Determine date plus home and away final scores
        h2s = str(h2).split()
        date = h2s[0] ; agoal = h2s[1] ; hgoal = h2s[3]
        date = date[4:].rstrip()
        agoal = int(agoal[5:].rstrip())
        hgoal = int(hgoal[:-5].lstrip())
    except:
        print("ERROR: Something is wrong parsing", url, "with headers h1:", h1, "h2:", h2)
        exit()

    return date, ateam, hteam, agoal, hgoal

def ParseNaturalStatTrickTables(df, soup, url, gindex, stats, sufst, sufsa, sufpp, sufpk, \
                                tabstid="tbts5v5", tabsaid="tbtssva", tabppid="tbtspp", tabpkid="tbtspk"):
    """
    Read standard (5v5), power-play, and penaltykill tables from naturalstattrick
    """

    # Collect standard (5v5), power-play, and penalty-kill tables for the home team
    dfst = ReadHTMLTable(soup, tabstid, url).loc[0]
    dfsa = ReadHTMLTable(soup, tabsaid, url).loc[0]
    dfpp = ReadHTMLTable(soup, tabppid, url).loc[0]
    dfpk = ReadHTMLTable(soup, tabpkid, url).loc[0]

    # Flag to indicate if data could not be found
    flagged = False

    for j in range(stats.shape[0]):
        stat = stats[j]
        statst = stat+sufst ; statsa = stat+sufsa ; statpp = stat+sufpp ; statpk = stat+sufpk
        try: df.loc[gindex,statst] = float(dfst[stat].split()[-1])
        except: flagged = True
        try: df.loc[gindex,statsa] = float(dfsa[stat].split()[-1])
        except: flagged = True
        try: df.loc[gindex,statpp] = float(dfpp[stat].split()[-1])
        except: flagged = True
        try: df.loc[gindex,statpk] = float(dfpk[stat].split()[-1])
        except: flagged = True

    return df, flagged

def PrintFMT(string, values=None, sl=40, fl=16):
    """
    Format output string.
    """

    s = string.ljust(sl)
    if values is not None:
        s += " : "
        for v in values: 
            s += str(v).ljust(fl)
    print(s)

def PrintSummaryStatistics(df, stats, sl=40, sufst="_5v5", sufsa="_SVA", sufpp="_PP", sufpk="_PK"):
    """
    Print some summary statistics and perform some consistency checks.
    """

    # Determine number of teams and verify that Home and Away columns agree on this
    teamsh = np.sort(df.Home.unique())
    teamsa = np.sort(df.Away.unique())
    assert np.array_equal(teamsh, teamsa)
    nteams = teamsh.shape[0]
    PrintFMT("Number of teams", (nteams,))

    # Determine the number of games played in the regular season for each team; verify same for all teams
    ngames = np.zeros(nteams, dtype="int32")
    for i in range(nteams): 
        ngames[i] = df.loc[((df.Home == TeamFMT(teamsh[i])) | (df.Away == TeamFMT(teamsh[i]))) & (df.Season == 0)].shape[0] 
    ngames = np.unique(ngames) ; assert ngames.shape[0] == 1
    PrintFMT("Number of regular season games per team", (ngames[0],))

    # Determine the total number of regular season and playoff games
    nregs    = df.loc[df.Season == 0].shape[0]
    nregs_na = nregs - df.dropna().loc[df.Season == 0].shape[0] 
    PrintFMT("Total number of regular season games", (nregs,))
    PrintFMT("Regular season games with missing data", (nregs_na,))
    nplays = df.loc[df.Season > 0].shape[0]
    nplays_na = nplays - df.dropna().loc[df.Season > 0].shape[0]
    PrintFMT("Total number of playoff games", (nplays,))
    PrintFMT("Playoff games with missing data", (nplays_na,))

    # Print stats for each attribute
    dfc = df._get_numeric_data()
    for stat in dfc:
        dc = dfc[stat]
        PrintFMT(stat, (dc.min(), dc.max(), dc.mean()), sl=10)

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

if not os.path.isdir(data_root_dir): os.makedirs(data_root_dir)

#
# Generate data one season at a time
#

for season in seasons:

    print("EXTRACTING DATA FOR " + str(season-1) + "-" + str(season) + " SEASON")

    # Start by extracting game information (teams, final scores) for all regular season and playoff games 
    df = ExtractSeasonGames(season)

    # Extract detailed shot information for each of these games 
    df = ExtractShotStatistics(df, shotstats, season)

    # Print summary statistics
    PrintSummaryStatistics(df, shotstats, season)

    # Save dataframe to CSV file 
    SaveData(df, data_root_dir, data_template, season)

    print()

