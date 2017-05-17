#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 15:17:55 2017

@author: fosa
"""

import random
import pandas as pd
import numpy as np
from itertools import chain



#because there are so many data points we can add
#we will go with the dynamic inserting

#there are many ways we can format the data
#we will re-organize the simplest way first
#each training example consists of 2 k_hots for team IDs (currently turned off)
#plus one bool each for home or away


def insertInResults(stats_array, results, fields_list):
    
    for i in range(len(results)):
        team_a = results.loc[i, "Team A"]
        team_b = results.loc[i, "Team B"]
        week = results.loc[i, "Week"]

        if week > 1:         
            for item in fields_list:
                temp_1 = (stats_array.loc[(stats_array["Week"] == week - 1) \
                     & (stats_array["Team"] == team_a), item].values)
                
                results.loc[i, "A " + item] = temp_1
                temp_2 = (stats_array.loc[(stats_array["Week"] == week - 1) \
                     & (stats_array["Team"] == team_b), item].values)

                results.loc[i, "B " + item] = temp_2
    return results
    
def add_Home(panda):

    for i in range(len(panda)):

        home_letter = panda.loc[i, "Home"]
   
        if (home_letter == "A"):
            panda.loc[i, "A Home"] = 1
            panda.loc[i, "B Home"] = 0

        if (home_letter == "B"):
            panda.loc[i, "A Home"] = 0
            panda.loc[i, "B Home"] = 1
        if (home_letter == "N"):
            panda.loc[i, "A Home"] = 0
            panda.loc[i, "B Home"] = 0



    return panda

def insertStatPF_PA(stats_array, year_results):
    
    current_week = 1
    weekly_stats = arrayInitiator(year_results)
    
    weekly_stats["Pts S"] = 0
    weekly_stats["Pts A"] = 0
    weekly_stats["# Games"] = 0
    
    last_week = year_results.iloc[-1]["Week"]
    stats_array.loc[:, "Pts S"] = 0
    stats_array.loc[:, "Pts A"] = 0
    
    
    for i in range(len(year_results)):
        if current_week != year_results.iloc[i]["Week"]:
            stats_array.loc[stats_array["Week"] == current_week, "Pts S"] = \
                weekly_stats["Pts S"]
            stats_array.loc[stats_array["Week"] == current_week, "Pts A"] = \
                weekly_stats["Pts A"]
            stats_array.loc[stats_array["Week"] == current_week, "# Games"] = \
                weekly_stats["# Games"]
                
            current_week += 1                
    
        pts_a = year_results.loc[i, "PtsA"]
        pts_b = year_results.loc[i, "PtsB"]
        team_a = year_results.loc[i, "Team A"]
        team_b = year_results.loc[i, "Team B"]
        
        weekly_stats.loc[weekly_stats["Team"] == team_a, "Pts S"] += pts_a
        weekly_stats.loc[weekly_stats["Team"] == team_a, "Pts A"] += pts_b
        weekly_stats.loc[weekly_stats["Team"] == team_b, "Pts S"] += pts_b
        weekly_stats.loc[weekly_stats["Team"] == team_b, "Pts A"] += pts_a
        
        weekly_stats.loc[weekly_stats["Team"] == team_a, "# Games"] += 1
        weekly_stats.loc[weekly_stats["Team"] == team_b, "# Games"] += 1
      
      

    return stats_array
    
def insertStatTO(stats_array, year_results):    
    current_week = 1
    weekly_stats = arrayInitiator(year_results)    
    weekly_stats["TO"] = 0
    stats_array.loc[:, "TO"] = 0 
    
    for i in range(len(year_results)):
        if current_week != year_results.iloc[i]["Week"]:
            stats_array.loc[stats_array["Week"] == current_week, "TO"] = \
                weekly_stats["TO"]                
            current_week += 1                    
        to_a = year_results.loc[i, "TOW"]
        to_b = year_results.loc[i, "TOL"]
        team_a = year_results.loc[i, "Team A"]
        team_b = year_results.loc[i, "Team B"]
        
        weekly_stats.loc[weekly_stats["Team"] == team_a, "TO"] += to_a
        weekly_stats.loc[weekly_stats["Team"] == team_b, "TO"] += to_b
    return stats_array
    
def insertStatYds(stats_array, year_results):    
    current_week = 1
    weekly_stats = arrayInitiator(year_results)    
    weekly_stats["Yds"] = 0
    stats_array.loc[:, "Yds"] = 0 
    
    for i in range(len(year_results)):
        if current_week != year_results.iloc[i]["Week"]:
            stats_array.loc[stats_array["Week"] == current_week, "Yds"] = \
                weekly_stats["Yds"]                
            current_week += 1                    
        to_a = year_results.loc[i, "YdsW"]
        to_b = year_results.loc[i, "YdsL"]
        team_a = year_results.loc[i, "Team A"]
        team_b = year_results.loc[i, "Team B"]
        
        weekly_stats.loc[weekly_stats["Team"] == team_a, "Yds"] += to_a
        weekly_stats.loc[weekly_stats["Team"] == team_b, "Yds"] += to_b
    return stats_array
        
    
def insertStatAvgPts(stats_array):
    stats_array["Avg Pts S"] = 0
    stats_array["Avg Pts A"] = 0
    
    for i in range(len(stats_array)):
        stats_array.loc[i, "Avg Pts S"] = stats_array.loc[i, "Pts S"] / stats_array.loc[i, "# Games"] / 23
        stats_array.loc[i, "Avg Pts A"] = stats_array.loc[i, "Pts A"] / stats_array.loc[i, "# Games"] / 23

    return stats_array
   
def insertStatDefeatedOpAvgS(stats_array, year_results):
    stats_array["D Op Avg Pts S"] = 0
  
    weekly_stats = arrayInitiator(year_results)
    weekly_stats["D Op Avg Pts S"] = 0
    stats_array.loc[:, "D Op Avg Pts S"] = 0
    
    for i in range(len(stats_array)):
        current_week = stats_array.loc[i, "Week"]

        if current_week > 1:
            current_team = stats_array.loc[i, "Team"]
            #on each entry from week 2 onwards check team a and team b
                    
            defeated_list = stats_array.loc[(stats_array["Week"] < current_week) & (stats_array["Team"] == current_team), "Defeated Team IDs"].tolist()
            defeated_list = [x for x in defeated_list if x != "posh"]
            d_op_avg_pts_s = []
            
            for team in defeated_list:
                d_op_avg_pts_s.append(stats_array.loc[(stats_array["Week"] == (current_week - 1)) & (stats_array["Team"] == team), "Avg Pts S"].values)    
            if len(d_op_avg_pts_s) > 0:
                stats_array.loc[i, "D Op Avg Pts S"] = np.sum(d_op_avg_pts_s) / len(d_op_avg_pts_s)

    return stats_array


    
    
def insertDefeatedOpAvgS(stats_array, year_results):

    for i in range(len(year_results)):
        current_week = year_results.loc[i, "Week"]
        
        if current_week > 1:
            team_a = year_results.loc[i, "Team A"]
            team_b = year_results.loc[i, "Team B"]
            #on each entry from week 2 onwards check team a and team b
            teams = [team_a, team_b]
            for k in range(len(teams)):
                if k == 0:
                    insert = "A D Op Avg Pts S"
                else:
                    insert = "B D Op Avg Pts S"
                    
                defeated_list = stats_array.loc[(stats_array["Week"] < current_week) & (stats_array["Team"] == teams[k]), "Defeated Team IDs"].tolist()
                defeated_list = [x for x in defeated_list if x != "posh"]
                d_op_avg_pts_s = []
                for team in defeated_list:
                    d_op_avg_pts_s.append(stats_array.loc[(stats_array["Week"] == (current_week - 1)) & (stats_array["Team"] == team), "Avg Pts S"].values)    
                if len(d_op_avg_pts_s) > 0:
                    year_results.loc[i, insert] = np.sum(d_op_avg_pts_s) / len(d_op_avg_pts_s)
             
    
    return year_results
    
def insertDefeatedOpAvgA(stats_array, year_results):

    for i in range(len(year_results)):
        current_week = year_results.loc[i, "Week"]
        
        if current_week > 1:
            team_a = year_results.loc[i, "Team A"]
            team_b = year_results.loc[i, "Team B"]
            #on each entry from week 2 onwards check team a and team b
            teams = [team_a, team_b]
            for k in range(len(teams)):
                if k == 0:
                    insert = "A D Op Avg Pts A"
                else:
                    insert = "B D Op Avg Pts A"
                    
                defeated_list = stats_array.loc[(stats_array["Week"] < current_week) & (stats_array["Team"] == teams[k]), "Defeated Team IDs"].tolist()
                defeated_list = [x for x in defeated_list if x != "posh"]
                d_op_avg_pts_s = []
                for team in defeated_list:
                    d_op_avg_pts_s.append(stats_array.loc[(stats_array["Week"] == (current_week - 1)) & (stats_array["Team"] == team), "Avg Pts A"].values)    
                if len(d_op_avg_pts_s) > 0:
                    year_results.loc[i, insert] = np.sum(d_op_avg_pts_s) / len(d_op_avg_pts_s)
        
    return year_results
        
def insertStatAvgTO(stats_array):
    stats_array["Avg TO"] = 0
    
    for i in range(len(stats_array)):
        stats_array.loc[i, "Avg TO"] = stats_array.loc[i, "TO"] / stats_array.loc[i, "# Games"]
        
    return stats_array
    

def insertStatAvgYds(stats_array):
    stats_array["Avg Yds"] = 0
    
    for i in range(len(stats_array)):
        stats_array.loc[i, "Avg Yds"] = stats_array.loc[i, "Yds"] / stats_array.loc[i, "# Games"] / 100
        
    return stats_array    
    
def insertStatPlayedTeams(stats_array, year_results):
     ######## unfinished
    stats_array["Defeated Team IDs"] = "posh"
    stats_array["Lost to Team IDs"] = "posh"

    #stats_array["Defeated Team IDs"] = stats_array["Defeated Team IDs"].astype(object)
    #stats_array["Lost to Team IDs"] = stats_array["Lost to Team IDs"].astype(object)
    """
    for i in range(len(stats_array)):
        stats_array.set_value(i, "Defeated Team IDs", [])
        array.set_value(i, "Lost to Team IDs", [])
    """ 
     
    for i in range(len(year_results)):
         
        pts_a = year_results.loc[i, "PtsA"]
        pts_b = year_results.loc[i, "PtsB"]
        team_a = year_results.loc[i, "Team A"]
        team_b = year_results.loc[i, "Team B"]
        current_week = year_results.loc[i, "Week"]     

        team_a_id = stats_array.loc[(stats_array["Team"] == team_a) & (stats_array["Week"] == current_week)].index.values
        team_b_id = stats_array.loc[(stats_array["Team"] == team_b) & (stats_array["Week"] == current_week)].index.values
     
    
        if year_results.loc[i, "Week"] == year_results.iloc[-1]["Week"]:
            return stats_array
    
        if pts_a > pts_b:
                stats_array.loc[team_a_id, "Defeated Team IDs"] = team_b
                stats_array.loc[team_b_id, "Lost to Team IDs"] = team_a
    
        if pts_a == pts_b:
                stats_array.loc[team_a_id, "Defeated Team IDs"] = team_b
                stats_array.loc[team_b_id, "Defeated Team IDs"] = team_a
                
    return stats_array
    
    
def insertStatByes(stats_array):
    stats_array["Byed"] = 0
    for i in range(len(stats_array)):
        if stats_array.loc[i, "Week"] > 1:
            current_week = stats_array.loc[i, "Week"]
            team = stats_array.loc[i, "Team"]
            previous_week_games_count = stats_array.loc[(stats_array["Team"] == team) & (stats_array["Week"] == current_week - 1), "# Games"] 
            current_week_games_count = stats_array.loc[(stats_array["Team"] == team) & (stats_array["Week"] == current_week), "# Games"]  
            if previous_week_games_count.values == current_week_games_count.values:
                stats_array.loc[i, "Byed"] = 1
    return stats_array
    
def addTeamIdentifier(year_results):
    tots = pd.read_csv("Datasets/NFL Results/totts2.csv")
    #get the value of the last week in results
    teams = pd.DataFrame(tots["Tm"].values, columns=["Team"])
    #team_id = np.zeros((1,32)).astype('f')
    alist = ["Team A", "Team B" ]
    home_list = ["A Home", "B Home"]
    for team in alist:
        for i in range(len(teams)):
            column_loc = year_results.columns.get_loc(home_list[alist.index(team)])
            year_results.insert(column_loc, (team +" id " + str(i+1)), 0 )
            #must insert this at the index where team A or team B is..
    for i in range(len(year_results)):
        team_a = year_results.loc[i, "Team A"]
        team_b = year_results.loc[i, "Team B"]
        
        team_a_id = teams.loc[teams["Team"] == team_a].index.values[0] + 1
        team_b_id = teams.loc[teams["Team"] == team_b].index.values[0] + 1
    
        year_results.loc[i, "Team A id " + str(team_a_id)] = 1
        year_results.loc[i, "Team B id " + str(team_b_id)] = 1
        
    return year_results
    
    
def insertLeagueAverage(stats_array):
    stats_array["League Avg Pts"] = 0
    max_weeks = stats_array.iloc[-1]["Week"]
    for i in range(max_weeks):
        #get and then set the whole week
        pts_mean = stats_array.loc[stats_array["Week"] <= (max_weeks - i), "Avg Pts S"].mean() 
        stats_array.loc[stats_array["Week"] == (max_weeks - i), "League Avg Pts"] = pts_mean
    
    return stats_array
    
def arrayInitiator(year_results):
    #this function creates a full 16 week table for each team
    #so we can store their data cumulatively
    tots = pd.read_csv("Datasets/NFL Results/totts2.csv") #build our team framework
    #get the value of the last week in results
    last_week = (year_results.iloc[-1]["Week"])
    
    results_array = pd.DataFrame(tots["Tm"].values, columns=["Team"])
    results_array["Week"] = 1
    
    for i in range((last_week - 2)): #setting the week values
        temp_comp = pd.DataFrame(tots["Tm"].values, columns=["Team"])
        temp_comp["Week"] = i + 2
        results_array = pd.concat([results_array, temp_comp])
        
    results_array = results_array.reset_index()
    del results_array["index"]
    return results_array


def insertStatNames(array, stats_list):
    #here we are inserting some columns into the full 16 week team table
    for name in stats_list:
        array[name] = 0       
    array["Defeated Team IDs"] = None
    array["Lost to Team IDs"] = None
    for i in range(len(array)):
        array.set_value(i, "Defeated Team IDs", [])
        array.set_value(i, "Lost to Team IDs", [])
    
        
    return array    

def prepareResultsFields(array, fields_list):
    for item in fields_list:    
        array.insert(2, "B " + item, 0 )
    
    for item in fields_list:
        array.insert(1, "A " + item, 0)

    return array
    
def delete_columns(array, columns):
    for column in columns:
        del array[column]
    return array
    
    
def prepareTrainingData(years_list):
    array_of_stats = []
    array_of_datas = []       
     
    for i in range(len(years_list)):
        
        nfl_results = pd.read_csv("Datasets/NFL Results/NFL results " + years_list[i] +".csv")
        print("year", years_list[i])
        nfl_cumulative_array = arrayInitiator(nfl_results)
        
        nfl_cumulative_array = insertStatNames(nfl_cumulative_array, stat_names_list)
        nfl_cumulative_array = insertStatPF_PA(nfl_cumulative_array, nfl_results)
        nfl_cumulative_array = insertStatTO(nfl_cumulative_array, nfl_results)
        nfl_cumulative_array = insertStatAvgTO(nfl_cumulative_array)
        nfl_cumulative_array = insertStatYds(nfl_cumulative_array, nfl_results)
        nfl_cumulative_array = insertStatAvgYds(nfl_cumulative_array)
        nfl_cumulative_array = insertStatPlayedTeams(nfl_cumulative_array, nfl_results)
        nfl_cumulative_array = (insertLeagueAverage(insertStatAvgPts(nfl_cumulative_array)))
        nfl_cumulative_array = insertStatDefeatedOpAvgS(nfl_cumulative_array, nfl_results)
        nfl_cumulative_array = insertStatByes(nfl_cumulative_array)
        
        array_of_stats.append(nfl_cumulative_array)
        
    array_of_stats = daisyChain(array_of_stats, stats_to_daisy)
    
    for i in range(len(years_list)):
        nfl_results = pd.read_csv("Datasets/NFL Results/NFL results " + years_list[i] +".csv")
        #print("year", years_list[i])
        nfl_cumulative_array = array_of_stats[i]
        
        nfl_results = delete_columns(nfl_results, columns_to_delete_from_imported_file)
        
        data_for_NN = prepareResultsFields(nfl_results, prep_list)  
        #data_for_NN = insertDefeatedOpAvgS(nfl_cumulative_array, data_for_NN)
        #data_for_NN = insertDefeatedOpAvgA(nfl_cumulative_array, data_for_NN)
        #data_for_NN = insertVictoriousOpAvgS(nfl_cumulative_array, data_for_NN)
        #data_for_NN = insertVictoriousOpAvgA(nfl_cumulative_array, data_for_NN)
        #temp_home = pd.read_csv("NFL 2015 home.csv").dropna().reset_index()
        data_for_NN = add_Home(data_for_NN)
        #data_for_NN = addTeamIdentifier(data_for_NN) #this should follow add HOME!!!
        data_for_NN = insertInResults(nfl_cumulative_array, data_for_NN, insert_list)
        #data_for_NN = delete_columns(pd.read_csv("NFL results " + year_list[i] +".csv"))
        array_of_datas.append(data_for_NN)

        
    return array_of_datas, array_of_stats


def daisyChain(all_stats_array, stats_to_daisy):
    #linking the last years data to the current year so that by the 70% done
    #point, we are using 100% of this years data
    #teams = pts_mean = stats_array.loc[stats_array["Week"] <= (max_weeks - i), "Avg Pts S"].mean()
    teams = all_stats_array[0].loc[all_stats_array[0]["Week"] == 1, "Team"]
    
    #make the stats for the current year, to be used on the next year
    
    for i in range(len(all_stats_array) - 1):
        last_week = all_stats_array[i].iloc[-1]["Week"]
        last_year_stats = all_stats_array[i]
        current_year_stats = all_stats_array[i + 1]
        
        #multiply each stat in each week for each team, by the descending ratio
        for k in range(1,10):
            print("week", k)
            for team in teams:
                #print("team", team)
                for stat in stats_to_daisy:
                    
                    previous_year_stat = last_year_stats.loc[(last_year_stats["Week"] == last_week) &
                                                            (last_year_stats["Team"] == team), stat].values[0]
                    current_year_stat = current_year_stats.loc[(current_year_stats["Week"] == k) & 
                                                            (current_year_stats["Team"] == team), stat].values[0]
                    changed_stat = (previous_year_stat * (1 - k/10.0)) + (current_year_stat * k/10.0) #stopping at week 10  
                    current_year_stats.loc[(current_year_stats["Week"] == k) & 
                                            (current_year_stats["Team"] == team), stat] = changed_stat
                    #print("Stat", stat,
                     #     "previous year", previous_year_stat,
                     #     "current year", current_year_stat,
                     #     "changed", changed_stat
                     #     )
            
    return all_stats_array
        
        
    
    

def calculateVegasSpreadError():
        
    df = pd.read_csv("Datasets/NFL Results/NFL results 2016 plus spread.csv")
    #239 games
    
    actual_spreads = []
    for i in range(239):
        if df.iloc[i]["Home"] == "A":
            actual_difference = df.iloc[i]["PtsA"] - df.iloc[i]["PtsB"]
        elif df.iloc[i]["Home"] == "B":
            actual_difference = df.iloc[i]["PtsB"] - df.iloc[i]["PtsA"]
        actual_spreads.append(actual_difference * -1) #negate it to get the format of a spread
    predicted_spreads = df.iloc[:239]["Home Spread"]
    
    actual_spreads = np.array(actual_spreads)
    
    absed = np.abs(predicted_spreads - actual_spreads)
    
    print("mean difference", np.mean(absed[:]))
    
    print("median difference", np.median(absed[:]))





#i'm thinking AVGs of these stats might not be so useful.  Something else.    

columns_to_delete_from_imported_file = ["YdsW", "YdsL", "TOL", "TOW"]



stats_to_daisy = ["Avg Yds", "Avg Pts S", "Avg Pts A", "D Op Avg Pts S", "Avg TO"]
insert_list = ["Avg Pts A", "Avg Pts S", "Avg TO", "Avg Yds", "D Op Avg Pts S", "Byed"]  #for rsults 
#insert_list = ["Avg Pts A", "Avg Pts S"] 
stat_names_list = ["PF", "PD"] #these are inserted for all the teams for each week
normalize_fields = ["Avg Pts A", "Avg Pts S"]#, "Avg TO"]#, "PD", "League Avg Pts"]
    
stat_names_list = ["PF", "PD"] #these are inserted for all the teams for each week
prep_list = ["Home", "Avg Pts A", "Avg Pts S", "Avg TO", "Avg Yds", "D Op Avg Pts S", "Byed"] #for results dataset
#prep_list = ["V Op Avg Pts A", "V Op Avg Pts S", "D Op Avg Pts A", "D Op Avg Pts S", "Avg TO", "Avg Pts A", "Avg Pts S", "Home"]  
 
years_list = ["2009", "2010","2011","2012","2013","2014", "2015","2016"]

all_years, all_stats = prepareTrainingData(years_list)


all_years_list = []
for i in range(len(years_list) - 1):
    all_years_list.append(all_years[i][16:])
combined = pd.concat(all_years_list)

combined.to_csv("Datasets/Training Data/NFL training set 2.csv")
all_years[-1][16:].to_csv("Datasets/Training Data/NFL validation set 2.csv")
