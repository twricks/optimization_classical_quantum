import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pulp import*


#Practicing Pandas and PuLP using reference CSVs previously utilized for the American Tech Fellowship, finding optimal vacation locations. 

np.random.seed(57)
bodies_of_water = True
parks = pd.read_csv("nationalparks.csv") #list of national parks
us_cities = pd.read_csv("uscities.csv") #list of US cities.
nature = pd.read_csv("natural_amenities_counties.csv") #US counties ranked by natural amenities. 
historical = pd.read_csv("nrhp.csv") #National Registry of Historic Places listings

#print("Parks:", parks.shape)
#print("Cities: ", us_cities.shape)
#print("Natural Amenities:", nature.shape)

#We wish to prune to parks near major cities, with the highest amenities. We'll begin by dropping cities outside a certain population size range, and counties with below a certain natural 
#amenity level. From the descriptive statistics of the ratings, we want to drop counties with a natural amenity below 5. 

mid_cities = us_cities[
    (us_cities['population'] >= 50_000) & # underscore for readability. Filtering with a goal of targeting midsize cities for convenience. 
    (us_cities['population'] <= 5000000)].copy()

print(f"Midsize cities kept: {len(mid_cities)}")

high_natural = nature[
    (nature['1=Low  7=High'] >= 4)  #mean in the data set is 3.5
].copy()

if bodies_of_water: #executes if the boolean is set to True. 
    high_natural = high_natural[high_natural['Water area pct'] >= 5].copy() #average in the data set is 4.5

print(f"Counties kept", {len(high_natural)})

high_historical = historical[
    (historical['Status'] == 'Listed') &
    (
        (historical['Level of Significance - National'] == True)
    ) &
    (historical['Category of Property'].isin(['DISTRICT', 'SITE'])) #isin() is a pandas method that checks whether a value is present in a list
]

print(f"Historic places kept: {len(high_historical)}")
#now we merge these two using the county FIPS:
#renaming the nature one's column
high_natural = high_natural.rename(columns={'combined FIPS': 'FIPS'})

mid_cities = mid_cities.merge(
    high_natural[['FIPS', '1=Low  7=High', 'Water area pct', 'County name', 'STATE']],
    left_on='county_fips',
    right_on='FIPS',
    how='inner' #this merge keeps only those cities where the FIPS matches
)


print("\nMerged cities with amenities:")
#print(mid_cities[['city', 'state_name', '1=Low  7=High']].head(10))

#we now have a set of 
#print(f"Midsize cities with good nature: {len(mid_cities)}")

historic_counties = set(high_historical['County'].dropna().unique())

# Keep only cities whose county appears in the historic places data
mid_cities = mid_cities[mid_cities['county_name'].isin(historic_counties)].copy()

print(f"Midsize cities kept after historic county filter: {len(mid_cities)}")

city_latitudes = mid_cities['lat'].values
city_longitudes = mid_cities['lng'].values

#midcities and parks are imported as dataframes. .values is a pandas feature that exports pandas series as numpy arrays.
park_latitudes = parks['latitude'].values
park_longitudes = parks['longitude'].values

max_allowed_dist = 200 #we want to drop parks with excessive distance from our verdant cities

good_parks = [] #creating a list to store the parks we keep

for i in range(len(parks)):
    lat_d = park_latitudes[i] - city_latitudes #numpy broadcasting
    lon_d = park_longitudes[i] - city_longitudes
    distances = np.sqrt(lat_d**2 + lon_d**2)
    min_dist_deg = distances.min()
    min_dist_miles = min_dist_deg * 69 #rough degrees-to-miles conversion
    if min_dist_miles <= max_allowed_dist:
# Keep a copy of this park row + add the distance
        park_row = parks.iloc[i].copy()
        park_row['min_dist_to_good_city_miles'] = min_dist_miles
        park_row['score'] = 1 /((min_dist_miles +10)) #creating a value score based on these distances, so that distant parks are less valuable. 
        good_parks.append(park_row)

#now we need to convert good_parks back into a data frame
better_parks = pd.DataFrame(good_parks)

#print(better_parks[['details', 'min_dist_to_good_city_miles']].head(10))
#print(f"Parks kept: {len(better_parks)}")

#because the linear combinatorial optimization problem will become very large for other approaches, we will randomly select a starting city,
#and look for the best combination of parks near it, several times. 

best_solutions = []
best_score = -np.inf
best_city = None
num_runs = 500

for run in range(num_runs):
    startingcity = mid_cities.sample(1).iloc[0] #selects a random row from the mid_cities using the random numpy seed, converts it from a Pandas series
    start_lat = startingcity['lat']
    start_lon = startingcity['lng']

    better_parks['dist_to_this_city'] = (
        np.sqrt(
            (better_parks['latitude'] - start_lat)**2 + 
            (better_parks['longitude'] - start_lon)**2
        ) * 69
    )
    
    #Filter parks reasonably close to THIS city
    nearby_parks = better_parks[better_parks['dist_to_this_city'] <= 300].copy()
    
    if len(nearby_parks) < 4:
        continue  # Skip if too few parks near this city
    
    nearby_parks = nearby_parks.reset_index(drop=True)
    n = len(nearby_parks)


    prob = LpProblem(f"Best_Parks_from_{startingcity['city']}", LpMaximize)

    x = LpVariable.dicts("select", range(n), cat="Binary")
    
    # Objective
    prob += lpSum([x[i] * nearby_parks.loc[i, 'score'] for i in range(n)])
    
    # Constraints
    prob += lpSum([x[i] for i in range(n)]) >= 3
    prob += lpSum([x[i] for i in range(n)]) <= 5
    prob += lpSum([x[i] * nearby_parks.loc[i, 'dist_to_this_city'] for i in range(n)]) <= 1200
    
    status = prob.solve(PULP_CBC_CMD(msg=False))
    
    if LpStatus[status] != 'Optimal':
        continue
    
    current_score = value(prob.objective)
    selected_idx = [i for i in range(n) if value(x[i]) > 0.5]
    selected_parks = nearby_parks.loc[selected_idx].copy()
    
    # Store this solution
    best_solutions.append({
        'score': current_score,
        'city': startingcity['city'],
        'state': startingcity['state_name'],
        'parks': selected_parks,
        'num_parks': len(selected_parks)
    })

#printing result
best_solutions.sort(key=lambda x: x['score'], reverse=True)


min_city_distance = 100   # Solutions cannot be too close to one another, are likely reflecting the same attractions. 

diverse_solutions = []
for sol in best_solutions:
    too_close = False
    for kept in diverse_solutions:
        # Calculating distance between starting cities, replacing previous solutions with higher scoring option where they are close. 
        lat1, lon1 = mid_cities[mid_cities['city'] == sol['city']].iloc[0][['lat', 'lng']]
        lat2, lon2 = mid_cities[mid_cities['city'] == kept['city']].iloc[0][['lat', 'lng']]
        dist = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 69
        
        if dist < min_city_distance:
            too_close = True
            break
    
    if not too_close:
        diverse_solutions.append(sol)

best_solutions = diverse_solutions   # Replace with the filtered list
print(f"\n=== Top {min(5, len(best_solutions))} Solutions ===\n")

for rank, sol in enumerate(best_solutions[:10], 1):
    print(f"{rank}. From: {sol['city']}, {sol['state']}")
    print(f"   Total Score: {round(sol['score'], 4)}")
    print(f"   Parks selected: {sol['num_parks']}")
    print("   Parks:")
    
    clean_parks = sol['parks'][['details', 'dist_to_this_city', 'score']].copy()
    clean_parks = clean_parks.reset_index(drop=True)
    print(clean_parks.to_string(index=False))
    print("-" * 60)




n_to_plot = 10   #Number of solutions to plot

plt.figure(figsize=(13, 9))

# Plot all midsize cities in the background
plt.scatter(mid_cities['lng'], mid_cities['lat'], 
            color='lightgray', s=10, alpha=0.4, label='All Midsize Cities')


colors = cm.get_cmap('tab10', n_to_plot)

for idx, sol in enumerate(best_solutions[:n_to_plot]):
    color = colors(idx)
    
    # Get the starting city row
    start_row = mid_cities[mid_cities['city'] == sol['city']].iloc[0]
    
    
    plt.scatter(start_row['lng'], start_row['lat'], 
                color=color, s=220, marker='*', edgecolors='black', 
                linewidths=0.8, zorder=5, label=f"{idx+1}. {sol['city']}")
    
    # add text label next to the star
    plt.text(start_row['lng'] + 0.8, start_row['lat'] + 0.3, 
             f"{idx+1}. {sol['city']}", fontsize=9, fontweight='bold')

plt.title(f"Top {n_to_plot} Solutions — Geographic Spread", fontsize=14)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(title="Starting City (Top Solutions)", 
           bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()