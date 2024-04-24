import pandas as pd
from pybaseball import playerid_lookup
from pybaseball import statcast_pitcher
from pybaseball import standings
from pybaseball import schedule_and_record
import matplotlib.pyplot as plt

reese = playerid_lookup('olson', 'reese')
reese = statcast_pitcher('2024-03-28', '2024-04-16', 681857)

reese = reese.loc[reese["game_date"] == '2024-04-15']

import seaborn as sns
sns.barplot(x = 'pitch_type', y = 'delta_run_exp', data = reese)
reese.groupby("pitch_type").agg("count")

reese['pitch_type'].value_counts()

from pybaseball import top_prospects

top_prospects(teamName = None, playerType = "batters")
from pybaseball import statcast_pitcher

reese = statcast_pitcher('2024-04-01', '2024-04-16', 681857)
sns.scatterplot(x = "release_pos_x", y = "release_pos_z", data = reese, hue = "pitch_type")

tarikid = playerid_lookup('corbin', 'patrick')
tarik = statcast_pitcher('2015-01-01', '2023-10-01', 571578)
sns.scatterplot(x = "release_pos_x", y = "release_pos_z", data = tarik, hue = "pitch_type")
plt.xlabel("Release Point X-Axis")
plt.ylabel("Release Point Z-Axis")
plt.title("Tarik Skubal 2024")

hunterid = playerid_lookup('verlander', 'justin')
hunter = statcast_pitcher('2024-03-28', '2024-04-16', 686613)

sns.scatterplot(x = "release_pos_x", y = "release_pos_z", data = hunter, hue = "game_date")
plt.xlabel("Release Point X-Axis")
plt.ylabel("Release Point Z-Axis")
plt.title("Hunter Brown 2024")

tarik.groupby("pitch_type").release_spin_rate.agg("mean")

tarikfastball = tarik.loc[tarik["pitch_type"] == "FF"]
tarikfastball

sns.scatterplot(x = "release_spin_rate", y = "delta_run_exp", data = tarikfastball)

tarikpitch = tarik.loc[tarik["pitch_type"] == "FF"]
tarikpitchinplay = tarikpitch.loc[tarikpitch["description"] == "hit_into_play"]

sns.scatterplot(x = "zone", y = "estimated_woba_using_speedangle", data = tarikpitchinplay)
tarikpitchinplay.groupby("zone").estimated_woba_using_speedangle.agg("mean")

434378


from pybaseball import pitching_stats
from pybaseball import batting_stats
from pybaseball import pitching_stats_bref
all = batting_stats(2024, qual = 1)
tigers = all.loc[all["Team"] == "DET"]
tigers

sns.scatterplot(x = "Clutch", y = "PA", data = tigers)
tigers['slgdiff'] = tigers['SLG'] - tigers['xSLG']
tigers = tigers.sort_values(by = 'slgdiff')


sns.barplot(x = 'Name', y = 'slgdiff', data = tigers, hue = 'PA', palette = 'viridis')
plt.ylabel("SLG - xSLG")
plt.title("2024 Tigers - Slugging vs. Expected Slugging")
plt.xticks(fontsize = 7, rotation = 32)


nobuddy = tigers.loc[tigers['PA'] > 10]
for index, row in nobuddy.iterrows():
    plt.text(row['SLG'], row['xSLG'], row['Name'], fontsize=8)
sns.scatterplot(x = 'SLG', y = 'xSLG', data = nobuddy, hue = 'PA', palette = 'viridis')
plt.title("2024 Tigers - Slugging vs. Expected Slugging")
plt.plot([nobuddy['SLG'].min(), nobuddy['SLG'].max()], [nobuddy['SLG'].min(), nobuddy['SLG'].max()], color='gray', linestyle='--')
plt.savefig('Tigers.png', dpi=300, bbox_inches='tight')







all = batting_stats(2024, qual = 1)


all['slgdiff'] = all['SLG'] - all['xSLG']
all = all.sort_values(by = 'slgdiff')




all1 = all.loc[all['PA'] > 20]
for index, row in nobuddy.iterrows():
    plt.text(row['SLG'], row['xSLG'], row['Name'], fontsize=8)
sns.scatterplot(x = 'SLG', y = 'xSLG', data = all1, hue = 'PA', palette = 'viridis')
plt.title("2024 MLB - Slugging vs. Expected Slugging")
plt.plot([nobuddy['SLG'].min(), nobuddy['SLG'].max()], [nobuddy['SLG'].min(), nobuddy['SLG'].max()], color='gray', linestyle='--')
plt.savefig('MLB.png', dpi=300, bbox_inches='tight')










