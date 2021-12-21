
"""
Satic Parking Allocation Using Gurobi for Different Vehicle Types
Dissertation - Part 1
Created on Mon Sep  7 17:46:21 2020

@author: akism
"""

#Import libraries: Gurobi solver, Pandas, datetime
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum as qs
import pandas as pd
import datetime

"""
Reading input files >

timeMatrix: A time matrix of travel times in minutes between origins, parking, 
and buildings.

streetMatrix: An incidence matrix for parking spots within each street.

info: Arriving vehicles information (destination, value of time, etc.)
"""

timeMatrix = pd.read_csv("C:\\Users\\akism\\OneDrive\\Desktop\\GitHub\\Dissertation_Static\\time.csv")
timeMatrix = timeMatrix.set_index("Node")
streetMatrix = pd.read_csv("C:\\Users\\akism\\OneDrive\\Desktop\\GitHub\\Dissertation_Static\\streets.csv")
streetMatrix = streetMatrix.set_index("street") 
info = pd.read_csv("C:\\Users\\akism\\OneDrive\\Desktop\\GitHub\\Dissertation_Static\\Info.csv")


"""
Defining Sets & Parameters
"""

#Get the number of origins (origins start have prefix "i")
originCount = 0
for i in  range(timeMatrix.shape[0]):
    if timeMatrix.index[i][0] == "i":
        originCount += 1  
        
#Get number of parking spaces from the parking-street incidence matrix       
parkingCount = streetMatrix.shape[1]

#Get the count of vehicles, buildings, service and delivery trucks, buses, and passenger cars
buildingCount =  timeMatrix.shape[0] - originCount - parkingCount
carCount = info.apply(lambda x: True if x['Vehicle'][:3] == "car" else False, axis =1).sum()
serviceCount = info.apply(lambda x: True if x['Vehicle'][:7] == "service" else False, axis =1).sum()
busCount = info.apply(lambda x: True if x['Vehicle'][:3] == "bus" else False, axis =1).sum()
truckCount = info.apply(lambda x: True if x['Vehicle'][:5] == "truck" else False, axis =1).sum()

#Defining garages and their capacities
garages = ["j361"]
garageCapacity = {"j361":50}

#Defining on-street parking capacity
onStreetCapacity = 1

#Defining VOT and flow rate per minute
averageVoT = 87/60 #per minute
laneFlowRate = 10 #per minute


#Constructing sets for Gurobi iterators
origins = ["i" + str(i+1)  for i in range(originCount)]
parks = ["j" + str(j+1)  for j in range(parkingCount)]
buildings = ["k" + str(k+1)  for k in range(buildingCount)]
nodes= origins + parks + buildings
streets = ["s" + str(i+1) for i in range(streetMatrix.shape[0])]
passCars = ["car" + str(v+1) for v in range(carCount)]
trucks = ["truck" + str(v+1) for v in range(truckCount)]
services = ["service" + str(v+1) for v in range(serviceCount)]
buses = ["bus" + str(v+1) for v in range(busCount)]

#Constructing sets combinations
passNservice = passCars + services
nonBuses = passCars + trucks + services
allTrucks = trucks + services
allVeh = passCars + trucks + services + buses
nonPassCars = trucks + services + buses
nonTrucks = passCars + services + buses
restrictedVehs = nonPassCars.copy()
restrictedTrucks = allTrucks.copy()
restrictedParking = garages.copy()

"""
Defining dictionnaries for parameters and sets for Gurobi usage
"""

#Origins
originI=[]
for index, row in info.iterrows():    
    originI.append("i" + str(row["Origin"]))
info["Supply"] = originI

#Pair-wise time matrix values
time={(i, j):timeMatrix.loc[i][j] for i in nodes for j in nodes}

#Driving VOT for each vehicle
VoT = {}
for index, row in info.iterrows():    
    VoT[row["Vehicle"]] = row["VoT"]/60
    
#Walking VOT for each vehicle
VoTWalk = {}
for index, row in info.iterrows():    
    if "bus" in row["Vehicle"]:
        VoTWalk[row["Vehicle"]] = 17.81/60
    else:
        VoTWalk[row["Vehicle"]] = row["VoT"]/60

#Parking duration for each vehicle
duration = {}
for index, row in info.iterrows():    
    duration[row["Vehicle"]] = row["Duration"]

#Parking spaces' capacities      
parkCapacity = {}
for j in parks:
    if j in garages:
        parkCapacity[j] = garageCapacity[j]
    else:
        parkCapacity[j] = onStreetCapacity
      
#Origins of each vehicle 
supply = {}
for i in origins:
    for v in allVeh:
         tup = i,v
         supply[tup] = 0
for index, row in info.iterrows():
    tup = row["Supply"],row["Vehicle"]
    supply[tup] = 1
    
 #Destination of each vehicle
destination = {}
for veh in nonBuses:
    destination[veh] = "k" + str(info.loc[info['Vehicle'] == veh,'Dest_start'].iloc[0])

#Truck-end destination for each vehicle
truckEndDestination = {}    
for truck in trucks:
    truckEndDestination[truck] = "k" + str(info.loc[info['Vehicle'] == truck,'Dest_end'].iloc[0])

#Demand of each destination for each vehicle
demand={(v,k): 0 for v in allVeh for k in buildings} 
totalDemand = {}
for v in nonBuses:
    totalDemand[v] = 1
    tup = v,destination[v]
    demand[tup]=1

#Setting up bus riders demand for each building 
busDestination = {}
busTotalDestination = {}
for index, row in info.iterrows():
    busDestinationList = []
    busDemandList = []
    if "bus" in row["Vehicle"]: 
        busDestination[row["Vehicle"]] = {}      
        busDestinationList = row["Dest_start"].split(",")
        busDemandList = row["Demand"].split(",")
        totalDemand[row["Vehicle"]] = 0
        for i in range (len(busDestinationList)):
           busDestination[row["Vehicle"]]["destination" + str(i + 1)] = "k"  + busDestinationList[i]
           tup = row["Vehicle"], "k"  + busDestinationList[i]
           demand[tup] = int(busDemandList[i])
           totalDemand[row["Vehicle"]] +=  int(busDemandList[i])
           busTotalDestination[row["Vehicle"]] = len(busDestinationList)
           
#Pair-wise incidence of parking-street combinations
streetIncidence = {}
for s in streets:
    for j in parks:
        tup = j, s
        streetIncidence[tup] = streetMatrix.loc[s][j]
        

'''
Setting up the Gurobi model.
The full model details are explained within the dissertation document
'''
#Initializing Gurobi model       
m = gp.Model()

#Defining Gurobi variables

x = m.addVars(origins,parks, allVeh ,vtype=GRB.BINARY, name="x")
y = m.addVars(parks,buildings,allVeh,vtype=GRB.BINARY, name="y")
yr = m.addVars(buildings,parks,allVeh,vtype=GRB.BINARY, name="yr")
occ = m.addVars(parks,allVeh ,vtype=GRB.BINARY, name="occ")
ex = m.addVars(parks,allVeh ,vtype=GRB.BINARY, name="ex")
pi = m.addVars(streets, vtype=GRB.INTEGER, name="pi",lb=0)

#Defining Constraints
m.addConstrs((x.sum(i,'*',v) == supply[i,v] for v in allVeh for i in origins), name="supply_origin")
m.addConstrs((x.sum('*',j,v) == occ[j,v] + ex[j,v] for j in parks for v in allVeh ), name="parking_occupancy")
m.addConstrs((occ.sum(j,'*') <= parkCapacity[j] for j in parks ), name="parking_capacity")
m.addConstrs((occ.sum(j,'*') >= ex.sum(j,'*') for j in parks), name="parking_capacity_maybe")
m.addConstrs((ex.sum(j,'*') <= 1 for j in parks ), name="double_park")
m.addConstrs((ex.sum(j,'*')  == 0  for j in garages) , name="Garage_no_double_park")
m.addConstrs((x.sum('*',j,v) == 0 for j in restrictedParking for v in restrictedVehs) , name = "restrcited_spaces_1")
m.addConstrs((y.sum(j,'*',v) == x.sum('*',j,v) for v in nonBuses for j in parks), name="const_3")
m.addConstrs((y.sum(j,'*',v) - busTotalDestination[v]*x.sum('*',j,v) == 0 for v in buses for j in parks), name="const_3")
for bus_id, bus_dest in busDestination.items():
    for stop in bus_dest:
         m.addConstr((y.sum('*',bus_dest[stop],bus_id) == 1) , name="demand_staisfaction")
         m.addConstr((yr.sum(bus_dest[stop],'*',bus_id) == 1) , name="demand_staisfaction")       
m.addConstrs((y.sum('*',k,v) == demand[v,k] for k in buildings for v in nonBuses), name="const_3b") #change into bus (normalized) and others
m.addConstrs((yr.sum(k,'*',v) == demand[v,k] for k in buildings for v in passNservice), name="const_3c") #change into bus (normalized) and others
for v in trucks:
    for j in parks:
        m.addConstr((y[j,destination[v],v] == yr[truckEndDestination[v],j,v]), name="const_6")
m.addConstrs((y[j,k,v] == yr[k,j,v] for j in parks for k in buildings for v in nonTrucks),  name="const_7") #check how to use sets eithout some elements
m.addConstrs((y[j,k,v] == yr[k,j,v] for j in parks for k in buildings for v in buses),  name="const_9")
for q in streets:
    for v in allVeh:
        m.addConstr((pi[q] >= duration[v]*qs(streetIncidence[j,q]*ex[j,v] for j in parks)),name = "street_incidence")

#Defining Objective Function
m.setObjective((qs(VoT[v]*time[i,j]*x[i,j,v]  for i in origins for j in parks for v in allVeh)+ \
                qs(VoTWalk[v]*time[j,k]*y[j,k,v]*demand[v,k]  for j in parks for k in buildings for v in allVeh)+ \
                qs(VoTWalk[v]*time[k,j]*yr[k,j,v]*demand[v,k]  for j in parks for k in buildings for v in allVeh)+ 
                qs(pi[q]*averageVoT*laneFlowRate for q in streets)),GRB.MINIMIZE)

#Solve
start = datetime.datetime.now()
m.optimize()
runTime= (datetime.datetime.now()-start) 
print("Run Time: ", runTime)


"""
Outputting variable values for values > 0
"""
var_names = []
var_values = []
for var in m.getVars():
    if var.X > 0.05: 
        var_names.append(str(var.varName))
        var_values.append(var.X)       
dictValues = {"var_names":var_names, "var_values": var_values} 
dictValuesDataFrame = pd.DataFrame(dictValues)
dictValuesDataFrame.to_csv("C:\\Users\\akism\\OneDrive\\Desktop\\GitHub\\Dissertation_Static\\allVars.csv")        
cols =  parks.copy()
rows = ["Park Occupancy", "Double-Park Occupancy"]  
outputs = pd.DataFrame(index= rows, columns = cols)
garageOcc = {}
garageEx = {}
for j in garages:
    garageOcc[j] = []
    garageEx[j] = []
  
for var in var_names:   
    if var[:3] == "occ":
        if var.split("[")[1].split(",")[0] not in garages:
             outputs.loc["Park Occupancy"][var.split("[")[1].split(",")[0]] = var.split(",")[1][:-1]
        else:
             garageOcc[var.split("[")[1].split(",")[0]].append(var.split(",")[1][:-1])
    elif  var[:2] == "ex":
        if var.split("[")[1].split(",")[0] not in garages:
             outputs.loc["Double-Park Occupancy"][var.split("[")[1].split(",")[0]] = var.split(",")[1][:-1]
        else:
             garageEx[var.split("[")[1].split(",")[0]].append(var.split(",")[1][:-1])
for j in garages:
        outputs.loc["Park Occupancy"][j] =  garageOcc[j]
        outputs.loc["Double-Park Occupancy"][j] =  garageEx[j]            
outputs.to_csv("C:\\Users\\akism\\OneDrive\\Desktop\\GitHub\\Dissertation_Static\\outputs.csv")