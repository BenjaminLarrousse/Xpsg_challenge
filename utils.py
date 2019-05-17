# imports
import pandas as pd
import numpy as np
import xmltodict


# Events attribute in the F24 opta files
event_keys = ['@id', '@event_id', '@type_id', '@period_id', '@min', '@sec',
              '@player_id', '@team_id', '@outcome', '@x', '@y', '@keypass',
              '@timestamp', '@last_modified', '@version', 'Q']


# Function to find players who switch teams during the winter break
def moving_players():
	with open('Noms des joueurs et IDs - F40 - L1 20162017.xml') as fd:
	    players = xmltodict.parse(fd.read(), process_namespaces=True)
	
	players_stat = [ x['Player'] for x in players['SoccerFeed']['SoccerDocument']['Team']]

	players_stats = [(x['@uID'],x['Stat'],x['Name']) for i in range(len(players_stat)) for x in players_stat[i]]

	moving_players = []
	for player in range(len(players_stats)):
	    for x in players_stats[player][1]:
			if x['@Type'] == 'leave_date':
		   		if (x['#text'] >= '2017-01-01') & (x['#text'] <= '2017-01-31'):
					moving_players.append(players_stats[player][0][1:])
	return moving_players


# Function to calculate time played for each player in a particular game
def calculate_time_played(events):
	# type_id 34 is the lineup
    lineups = events[events['@type_id'] == '34']

    first_half_duration = max(events[events['@period_id'] == 1 ]['minutes'])
    second_half_duration = max(events[events['@period_id'] == 2 ]['minutes']) - 45

    game_duration = first_half_duration + second_half_duration
    
    result = pd.DataFrame()
    
    for index, row in lineups.iterrows():
        # find player in lineups for each team, with positions and jersey numbers
        min_dict= {'player_id': [x['@value'] for x in row['Q'] if x['@qualifier_id'] == '30'][0].split(', '),
                   'player_pos': [x['@value'] for x in row['Q'] if x['@qualifier_id'] == '44'][0].split(', '),
                   'player_jersey': [x['@value'] for x in row['Q'] if x['@qualifier_id'] == '59'][0].split(', '),
                   }

        min_played = pd.DataFrame.from_dict(min_dict)
        
        min_played.loc[min_played['player_pos'] != '5','minutes'] = game_duration
        
        min_played.loc[min_played['player_pos'] == '5','minutes'] = 0
        
        min_played.sort_values(by='player_id', inplace=True)
        
        # events send on (type_id = 19)
        temp_df = events[(events['@type_id'] == '19') &
                         (events['@team_id'] == row['@team_id'])][['@player_id', 'minutes', '@period_id']]
        
        temp_df.sort_values(by='@player_id', inplace=True)
        
        min_played.loc[min_played['player_id'].isin(temp_df['@player_id']),
                       'minutes'] = - temp_df['minutes'].values - (2 - temp_df['@period_id'].values
                                                                   ) * first_half_duration \
        + (temp_df['@period_id'].values-1) * 45 + second_half_duration
        
        # look at subtitutions events
        # events send off (type_id = 18) and force to go off (type_id = 20)
        temp_df = events[((events['@type_id'] == '18') | (events['@type_id'] == '20')
                          ) &
                         (events['@team_id'] == row['@team_id'])][['@player_id','minutes','@period_id']]
        
        temp_df.sort_values(by='@player_id', inplace=True)
        
        min_played.loc[min_played['player_id'].isin(temp_df['@player_id']),
                       'minutes'] = min_played.loc[min_played['player_id'].isin(temp_df['@player_id']),
                                                   'minutes'] \
        - (game_duration - (-(2 - temp_df['@period_id'].values) * temp_df['minutes'].values \
                            + (temp_df['@period_id'].values-1
                               ) * (first_half_duration + temp_df['minutes'].values - 45)
                            )
           )
        
        # events cards (type_id = 17)
        temp_df = events[(events['@type_id'] == '17') &
                 (events['@team_id'] == row['@team_id'])][['@player_id','minutes','Q','@period_id']]
    
        temp_df['red_cards'] = temp_df['Q'].apply(
            lambda x: sum([1 if y['@qualifier_id'] in ['32','33'] else 0 for y in x
                           ]))
        
        temp_df = temp_df[temp_df['@player_id'] != '']
        
        temp_df.sort_values(by='@player_id', inplace=True)
        
        if sum(temp_df['red_cards']) != 0:
            min_played.loc[min_played['player_id'].isin(temp_df[temp_df['red_cards'] > 0]['@player_id']),
                           'minutes'] = min_played.loc[min_played['player_id'].isin(
                                            temp_df[temp_df['red_cards'] > 0]['@player_id']),
                                            'minutes'] \
            - (game_duration - (-(2 - temp_df[temp_df['red_cards'] > 0]['@period_id'].values
                                  ) * temp_df[temp_df['red_cards'] > 0]['minutes'].values \
                                + (temp_df[temp_df['red_cards'] > 0]['@period_id'].values-1
                                   ) * (first_half_duration + temp_df[temp_df['red_cards'] > 0
                                                                      ]['minutes'].values - 45)
                                )
               )
        
        result = result.append(min_played, ignore_index=True)
        
    return result


# Function to calculate total game time for each player from a list of game files
def create_player_game_time(files, event_keys):
    tot_game_time = pd.DataFrame(columns=['player_id','minutes'])
    
    for file in files:
        with open(folder + '/' + file) as fd:    
            doc = xmltodict.parse(fd.read(), process_namespaces=True)
        
        events = pd.DataFrame(columns=event_keys)
        
        for x in doc['Games']['Game']['Event']:
            events.loc[len(events)] = [x[key] if key in x.keys() else '' for key in event_keys]
        
        events['minutes'] = (events['@min'].astype(int) * 60 + events['@sec'].astype(int)) / 60
        events['@period_id'] = events['@period_id'].astype(int)
        
        game_stat = calculate_time_played(events)

        for ind_sub, row_sub in game_stat.iterrows():
            if tot_game_time['player_id'].str.contains(row_sub['player_id']).any():
                tot_game_time.loc[tot_game_time['player_id'] == row_sub['player_id'],
                                  ['minutes']] += row_sub['minutes']
            else:
                tot_game_time = tot_game_time.append({'player_id': row_sub['player_id'],
                                                      'minutes': row_sub['minutes']
                                                      }, ignore_index=True)
        
        print('Game {} vs {} done'.format(doc['Games']['Game']['@home_team_name'],
                                          doc['Games']['Game']['@away_team_name']))
    return tot_game_time



# Function to append data from a list of Opta F24 files in a pandas dataframe
# Player_ids is a list of player ids from the F40 file (see create_player_game_time function)
def preprocessing(player_ids, files):
    result = pd.DataFrame()
    
    for file in files:
        with open(folder + '/' + file) as fd:    
            doc = xmltodict.parse(fd.read(), process_namespaces=True)
        
        events = pd.DataFrame(columns=event_keys)
        
        for x in doc['Games']['Game']['Event']:
            events.loc[len(events)] = [x[key] if key in x.keys() else '' for key in event_keys]
        events['game_id'] = doc['Games']['Game']['@id']
        events['@player_id'] = [int(x) if x != '' else 0 for x in events['@player_id']]
        events.loc[events['@team_id'] == doc['Games']['Game']['@home_team_id'],'@team_id'] = 1
        events.loc[events['@team_id'] == doc['Games']['Game']['@away_team_id'],'@team_id'] = 0
        
        result = result.append(events[events['@player_id'].isin(player_ids['player_id'])],
                               ignore_index=True)
        
        print(len(result))
        print('Game {} vs {} done'.format(doc['Games']['Game']['@home_team_name'],
                                          doc['Games']['Game']['@away_team_name']))
    return result


# Function to do data preparation on the dataframe we created with the preprocessing function
def preparation(df):
	res = df.copy()
	res['minutes'] = (res['@min'].astype(int) * 60 + res['@sec'].astype(int)) / 60
	res['@period_id'] = res['@period_id'].apply(lambda x: int(x))
	res['@keypass'] = [int(x) if ~np.isnan(x) else 0 for x in res['@keypass']]

	dummy_period = pd.get_dummies(res['@period_id'], prefix='period')

	dummy_team = pd.get_dummies(res['@team_id'], prefix='team')

	dummy_outcome = pd.get_dummies(res['@outcome'], prefix='outcome')

	feat_to_drop = ['@period_id', '@team_id', '@outcome',
		            '@id', '@event_id', '@timestamp', '@last_modified', '@version', '@min', '@sec',
		            'Q']

	res.drop(feat_to_drop, inplace=True, axis=1)

	res = pd.concat((res, dummy_period, dummy_team, dummy_outcome),
		           	axis=1)
	return res


# Function to create example ready to be fed to the Recurrent Neural network
# Code example of how to use it:
# games = df.groupby('game_id')
# examples = []
# labels = []
# ex_id = 0
#
# games.apply(lambda x: create_example(x, labels, ex_id, examples))
#
def create_example(group, labels, ex_id, examples):
    for i in range(1,3):
        half_max = max(group[group['period_' + str(i)] == 1]['minutes']) * 60

        for cut in range(45*60*(i-1),
                         45*60*(i-1) + int(half_max)+1 - 15*60,10):
            temp_df = group[(group['period_' + str(i)] == 1) &
                            (group['minutes'] >= cut/60) &
                            (group['minutes'] < cut/60 + 15)]
            
            temp_df.drop('game_id', inplace=True, axis=1)
            
            player_set = set([x for x in temp_df['@player_id'].unique() if x in players_labels['player_id'].values
                              ])
            
            for mi in player_set:
                # if mi in player_data['player_id'].values:
                labels.append((ex_id, mi))
                ex_id += 1
                example = temp_df.copy().reset_index(drop=True)
                example.loc[example['@player_id'] != mi,'@player_id'] = 0
                example.loc[example['@player_id'] == mi,'@player_id'] = 1
                example.loc[:len(example)-11,'@x'] = 0
                example.loc[:len(example)-11,'@y'] = 0
                example['delta_time'] = example['minutes'].diff()
                example.drop('minutes', inplace=True, axis=1)
                examples.append(example[example['@player_id'] == 1].to_numpy())

