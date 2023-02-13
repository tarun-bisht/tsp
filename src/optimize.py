import numpy as np
import pandas as pd
import folium
from pyomo.environ import (ConcreteModel, 
                           Var, 
                           ConstraintList, 
                           Objective, 
                           Binary, 
                           PositiveIntegers,
                           SolverFactory)


def initiate_solver(solver: str, bin_path: str):
    return SolverFactory(solver, executable=bin_path)


def get_map(cities_data, cities_distances, polylines=None, distances=None):
    cities_coords = cities_data.iloc[:, 2:4].to_numpy()
    cities_names = cities_data.iloc[:, 1].to_numpy()
    cities_dists = cities_distances.to_numpy()

    min_pt = (cities_data[['lat', 'lon']].min().values - 2).tolist()
    max_pt = (cities_data[['lat', 'lon']].max().values + 2).tolist()
    
    mymap = folium.Map(location=[28.5011226, 77.4099794])
    for name, pt in zip(cities_names, cities_coords):
        folium.Marker(list(pt), popup = name).add_to(mymap)
    mymap.fit_bounds([min_pt, max_pt])
    if polylines is not None:
        for i in range(len(polylines)-1):
            line = tuple([polylines[i], polylines[i+1]])
            if distances is not None:
                folium.PolyLine(line, tooltip=distances[i], popup=distances[i], weight=6, color="#111").add_to(mymap)
            else:
                folium.PolyLine(line, weight=0.5, color="#111").add_to(mymap)
    return mymap


def read_data(city_data: str, city_distances: str):
    cities_data = pd.read_csv(city_data, index_col=0)
    cities_distances = pd.read_csv(city_distances, index_col=0)
    cities_coords = cities_data.iloc[:, 2:4].to_numpy()
    cities_names = cities_data.iloc[:, 1].to_numpy()
    distance_matrix = cities_distances.to_numpy()
    return cities_coords, cities_names, distance_matrix


def TSPModel(distance_matrix):
    n = len(distance_matrix)
    model = ConcreteModel(name="TSPSolver")

    model.x = Var(range(n), range(n), initialize=0.0, domain=Binary)
    model.u = Var(range(n), domain=PositiveIntegers)

    model.objective = Objective(expr=sum(distance_matrix[i, j]*model.x[i, j] for i in range(n) for j in range(n)))

    model.constraints = ConstraintList()
    for i in range(n):
        model.constraints.add(sum(model.x[i, k] for k in range(n) if i != k) == 1)

    for j in range(n):
        model.constraints.add(sum(model.x[k, j] for k in range(n) if j != k) == 1)

    model.constraints.add(model.u[0] == 1)
    for i in range(1, n):
        model.u[i].setlb(2)
        model.u[i].setub(n)

    for i in range(n):
        for j in range(n):
            if i != 0 and j != 0:
                model.constraints.add(model.u[i] - model.u[j] + 1 <= (n-1)*(1-model.x[i, j]))
    return model


def parse_optimal_path_index(model: ConcreteModel):
    return [int(model.u[i].value) for i in range(len(model.u))]


def parse_optimal_path(model: ConcreteModel, cities_names: list):
    optimal_path_idx = parse_optimal_path_index(model)
    return  [cities_names[i-1] for i in optimal_path_idx]


def get_TSP_map(city_data: str, city_distances: str, model: ConcreteModel):
    cities_data = pd.read_csv(city_data, index_col=0)
    cities_distances = pd.read_csv(city_distances, index_col=0)
    cities_coords = cities_data.iloc[:, 2:4].to_numpy()
    cities_names = cities_data.iloc[:, 1].to_numpy()
    distance_matrix = cities_distances.to_numpy()
    optimal_path_idx = parse_optimal_path_index(model)

    polylines = [cities_coords.tolist()[i-1] for i in optimal_path_idx] + [cities_coords.tolist()[optimal_path_idx[0]-1]]
    distances = []
    for i in range(len(optimal_path_idx)-1):
        distances.append(f"{distance_matrix[optimal_path_idx[i]-1, optimal_path_idx[i+1]-1] :.2f} K.M.")
    distances.append(f"{distance_matrix[optimal_path_idx[-1]-1, optimal_path_idx[0]-1] :.2f} K.M.")
    
    return get_map(cities_data, cities_distances, polylines, distances)
