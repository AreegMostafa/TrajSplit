import pandas as pd
from tqdm import tqdm
import datetime
import math
import time
import requests
import pickle as pk
import os
import sys
import numpy as np 



def run(sta_path, file, results_path, agg_path, server):

    def getDistanceFromLatLonInm(lat1,lon1,lat2,lon2): 
        R = 6378100 # radius in meters
        deg_cov = math.pi/180
        dLat = (deg_cov)*(lat2-lat1)  # deg2rad below
        dLon = deg_cov*(lon2-lon1) 
        a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(deg_cov*(lat1)) * math.cos(deg_cov*(lat2)) * math.sin(dLon/2) * math.sin(dLon/2)
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)); 
        d = R * c # Distance in km
        return d

    def osrm_request(coords1, coords2, server):
        get_req = f'http://{server}:5000/route/v1/driving/{coords1};{coords2}?alternatives=false&steps=false&geometries=geojson&overview=false&annotations=false'
        req = requests.get(get_req)
        res = req.json()

        return res



    def calcBearing (lat1, long1, lat2, long2):
        dLon = (long2 - long1)
        x = math.cos(math.radians(lat2)) * math.sin(math.radians(dLon))
        y = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(dLon))
        bearing = math.atan2(x,y)   # use atan2 to determine the quadrant
        bearing = math.degrees(bearing)

        return bearing

    def calcNSEW(lat1, long1, lat2, long2):
        points = [('N', None), ('N', 'E'), (None, 'E'), ('S', 'E'), ('S', None), ('S', 'W'), (None, 'W'), ('N', 'W')]
        bearing = calcBearing(lat1, long1, lat2, long2)
        bearing += 22.5
        bearing = bearing % 360
        bearing = int(bearing / 45) # values 0 to 7
        NSEW = points [bearing]

        return NSEW

    def direction_heuristic(traj, M): 
        starting_loc = traj.head(1)[['Longitude', 'Latitude']].values[0]
        prev_loc = traj.head(1)[['Longitude', 'Latitude']].values[0]
        prev_time, tripid = traj.head(1)[['Position Date Time', 'trip_id']].values[0]
        
        hit = 0
        miss = 0
        counter = 0
        prev_cons_dist = 0
        prev_tot_speed = 0
        prev_tot_time_diff = 0

        trip_id = str(int(float(tripid)))+"-"+str(counter)
        
        trips_to_drop = []
        new_trip_heads = []
        prev_idx = None
        df = traj.copy()

        for idx, row in df.iterrows():
            time_diff = (row['Position Date Time']- prev_time) // datetime.timedelta(seconds=1) 

            cons_dist = getDistanceFromLatLonInm(prev_loc[1], prev_loc[0], row['Latitude'], row['Longitude'])
            global_direction = calcNSEW(starting_loc[1], starting_loc[0],  row['Latitude'], row['Longitude'])
            local_direction = calcNSEW(prev_loc[1], prev_loc[0],  row['Latitude'], row['Longitude'])
            if (global_direction[0] != local_direction[0] and (global_direction[0]!=None and local_direction[0]!=None)) or (global_direction[1] != local_direction[1] and (global_direction[1]!=None and local_direction[1]!=None)):
                try:
                    time.sleep(0.001)
                    res = osrm_request("{},{}".format(starting_loc[0], starting_loc[1]), "{},{}".format(row['Longitude'], row['Latitude']), server)
                    shortest_distance = res['routes'][0]['distance']

                    if(prev_cons_dist+cons_dist > shortest_distance*M):
                        if(prev_cons_dist < 20 or prev_tot_speed < 20 or prev_tot_time_diff < 100):
                            trips_to_drop.append(trip_id)

                        hit+=1
                        counter +=1

                        trip_id = str(int(float(tripid)))+"-"+str(counter)                    
                        trip_head = df.loc[prev_idx].copy()
                        trip_head.at['trip_id'] = trip_id
                        trip_head.at['distance'] = 0
                        trip_head_dict = trip_head.to_dict().copy()
                        new_trip_heads+=[trip_head_dict]

                        df.at[idx, "trip_id"] = trip_id
                        df.at[idx, "distance"] = cons_dist

                        prev_cons_dist = cons_dist 
                        prev_tot_speed = row['Average Speed']
                        prev_tot_time_diff = time_diff
                        starting_loc = trip_head['Longitude'], trip_head['Latitude']
                        prev_loc = row['Longitude'], row['Latitude']
                        prev_time = row['Position Date Time']
                        continue
                    else:
                        miss+=1
                        
                except Exception as e:
                    print(res, e)
                    pass
            prev_cons_dist+=cons_dist
            prev_loc = row['Longitude'], row['Latitude']

            prev_tot_speed += row['Average Speed']
            prev_tot_time_diff += time_diff
            prev_time = row['Position Date Time']
            prev_idx = idx

            df.at[idx, "trip_id"] = trip_id
            df.at[idx, "distance"] = cons_dist

        if(prev_cons_dist < 20 or prev_tot_speed < 20 or prev_tot_time_diff < 100):
            trips_to_drop.append(trip_id)

        
        return df, trips_to_drop, hit, miss, new_trip_heads



    traj = pd.read_csv(sta_path+file, compression="xz")
    traj.drop([c for c in traj.columns if "name" in c], axis=1, inplace=True)

    Ms = [0, 25, 50, 75, 100] 
    trips = traj.trip_id.unique()
    num_of_points = len(traj)

    list_of_properties = []

    for M in Ms:
        trip_heads = []
        temp = {}
        new_trajs = []
        sub_trip_heads = []
        e_failed = []
        t0 = time.time()
        hit = 0
        miss = 0
        for trip in tqdm(trips):
            trip_df = traj.query(f"trip_id == {trip}").copy()
            trip_df['trip_id'] = pd.Series(trip_df['trip_id'], dtype="string")

            trip_df['Position Date Time'] = pd.to_datetime(trip_df['Position Date Time'], format='%Y-%m-%d %H:%M:%S')
            new_traj, failed_trips, sub_hit, sub_miss, sub_trip_heads = direction_heuristic(trip_df, 1+(M/100))
            new_traj['prev_trip_id'] = trip

            sub_trip_heads = [h for h in sub_trip_heads if type(h) == dict]

            new_trajs.append(new_traj)

            duplicated_trip_heads = pd.DataFrame(sub_trip_heads)
            duplicated_trip_heads['prev_trip_id'] = trip

            trip_heads.append(duplicated_trip_heads.drop([c for c in duplicated_trip_heads.columns if "name" in c], axis=1))

            e_failed+=failed_trips
            hit+=sub_hit
            miss+=sub_miss
        t1 = time.time()

        new_trajs += trip_heads

        if len(new_trajs) > 1:
            endf = pd.concat(new_trajs, ignore_index= True).sort_values(by="Position Date Time").reset_index(drop=True)

            e_extra_seg = endf[(endf['trip_id'].isin(e_failed))]
            endf_without_short_trips = endf[~(endf['trip_id'].isin(e_failed))]
            num_oftrips = len(endf_without_short_trips['trip_id'].unique())
            time_taken = t1-t0

            temp['Num_of_points'] = num_of_points
            temp['time'] = time_taken
            temp['num_of_trips'] = num_oftrips
            temp['alpha'] = M
            temp['num_of_hits'] = hit
            temp['num_of_miss'] = miss
            temp['num_of_short_trips'] = len(e_extra_seg.trip_id.unique())
            temp['overall_distance'] = endf['distance'].sum() 
            temp['overall_distance_w_short'] = endf_without_short_trips['distance'].sum() 
            temp['baseline_number_of_trips'] = len(trips) 

            list_of_properties.append(temp)
            print(temp)

            md = f"{M}%"
            e_p = results_path + file+f"-Direction-{md}"
            endf.to_csv(e_p, compression="xz")

    with open(agg_path + file+f"-Direction-properties.pk", 'wb') as f:
        pk.dump(list_of_properties, f)
    print(list_of_properties)

