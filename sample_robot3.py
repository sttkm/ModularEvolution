import os

import time
import pickle
from collections import Counter
import numpy as np
import pandas as pd


def get_adjacent_points(module, module_key):
    module_pad = np.pad(module, 1)

    ### robotの隣接する空きvoxelの位置を探す
    r_i, r_j = np.where(module_pad > 0)
    adjacent_diff_i = np.array([-1, 0, 1, 0])
    adjacent_diff_j = np.array([0, 1, 0, -1])
    # 全てのvoxelの隣接voxelを調べる
    adjacent_i = np.expand_dims(r_i, axis=-1) + np.expand_dims(adjacent_diff_i, axis=0)
    adjacent_j = np.expand_dims(r_j, axis=-1) + np.expand_dims(adjacent_diff_j, axis=0)
    neighbors = module_pad[adjacent_i, adjacent_j]
    adjacent_points = pd.DataFrame(columns=["index", "point_i", "point_j", "direction_i", "direction_j", "module"], dtype="Int64")
    # 隣接voxelが空きである場所
    for r, d in zip(*np.where(neighbors == 0)):
        # # 隣接voxelの位置（pad解除）
        point = (adjacent_i[r, d] - 1, adjacent_j[r, d] - 1)
        idx = len(adjacent_points)
        adjacent_points.loc[idx] = [idx, point[0], point[1], -adjacent_diff_i[d], -adjacent_diff_j[d], module_key]
    
    return adjacent_points

def get_possible_coordinates(base_key, adjacents1, adjacents2):
    direction1 = adjacents1.loc[:, ["direction_i", "direction_j"]].values
    direction2 = adjacents2.loc[:, ["direction_i", "direction_j"]].values
    connection = np.all(np.expand_dims(direction1, axis=1) == -np.expand_dims(direction2, axis=0), axis=-1)

    possibles = {}
    for i1, row1 in adjacents1.iterrows():
        point1 = row1[["point_i", "point_j"]].values
        direction1 = row1[["direction_i", "direction_j"]].values
        module1 = row1["module"]
        for i2, row2 in adjacents2[connection[i1]].iterrows():
            point2 = row2[["point_i", "point_j"]]
            module2 = row2["module"]
            coordinate = tuple(point1 - point2 + direction1)
            edges = possibles.get(coordinate, [])
            pair = (i1, i2)
            edges.append(pair)
            possibles[coordinate] = edges

    coordinates = list(possibles.keys())
    possibles = pd.DataFrame({
        "coordinate_i": [c[0] for c in coordinates],
        "coordinate_j": [c[1] for c in coordinates],
        str(base_key): [possibles[c] for c in coordinates]
        }, columns=["coordinate_i", "coordinate_j", str(base_key)])
    return possibles

class Module:
    def __init__(self):
        self.key = None
        self.bias = None
        self.module = None
        self.adjacent_points = None

    def set_module(self, key, module):
        self.key = key
        self.module = module
        self.adjacent_points = get_adjacent_points(self.module, self.key)

    
def make_stable_modules():
    shapes = [
        np.array([[1, 1]], dtype=int),
        np.array([[1], [1]], dtype=int),
        np.array([[1, 1], [1, 0]], dtype=int),
        np.array([[1, 0], [1, 1]], dtype=int),
        np.array([[0, 1], [1, 1]], dtype=int),
        np.array([[1, 1], [0, 1]], dtype=int),
    ]
    modules = {}
    module_i = 1
    for t in [1, 3, 4]:
        for shape in shapes:
            module = shape.copy()
            module[module == 1] = t
            module_obj = Module()
            module_obj.set_module(module_i, module)
            modules[module_i] = module_obj
            module_i += 1
    
    module_n = len(modules)
    keys = list(modules.keys())
    possible_coordinates = {}
    for i in range(module_n):
        key1 = keys[i]
        module1 = modules[key1]
        for j in range(module_n):
            if i == j:
                continue
            key2 = keys[j]
            module2 = modules[key2]
            possible_coordinates[(key1, key2)] = get_possible_coordinates(key1, module1.adjacent_points, module2.adjacent_points)
    
    return modules, possible_coordinates





def connect_module(robot, module_map, module_coordinates, module, module_key, coordinate):
    shape_r = robot.shape
    shape_m = module.shape

    pad_i = (max(0, -coordinate[0]), max(0, coordinate[0] + shape_m[0] - shape_r[0]))
    pad_j = (max(0, -coordinate[1]), max(0, coordinate[1] + shape_m[1] - shape_r[1]))

    robot_ = np.pad(robot, (pad_i, pad_j))
    module_map_ = np.pad(module_map, (pad_i, pad_j))
    p_i = coordinate[0] + pad_i[0]
    p_j = coordinate[1] + pad_j[0]
    robot_[p_i:p_i+shape_m[0], p_j:p_j+shape_m[1]][module > 0] = module[module > 0]
    module_map_[p_i:p_i+shape_m[0], p_j:p_j+shape_m[1]][module > 0] = module_key
    module_coordinates["coordinate_i"] += pad_i[0]
    module_coordinates["coordinate_j"] += pad_j[0]
    module_coordinates.loc[module_key, ["coordinate_i", "coordinate_j"]] = [p_i, p_j]

    return robot_, module_map_, module_coordinates.astype("Int64")

def check_adjacent_validity(robot, module, coordinates, size_limit):
    shape_m = module.shape
    shape_r = robot.shape
    robot_ = np.pad(robot, ((shape_m[0], ), (shape_m[1], )))
    module_exists = np.where(module > 0, True, False)
    valid_coordinates = []
    for coordinate in coordinates:
        pad_i = (max(0, -coordinate[0]), max(0, coordinate[0] + shape_m[0] - shape_r[0]))
        pad_j = (max(0, -coordinate[1]), max(0, coordinate[1] + shape_m[1] - shape_r[1]))
        valid = (shape_r[0] + pad_i[0] + pad_i[1]) <= size_limit[0] and (shape_r[1] + pad_j[0] + pad_j[1]) <= size_limit[1]
        if not valid:
            continue
        i = coordinate[0] + shape_m[0]
        j = coordinate[1] + shape_m[1]
        robot_exists = np.where(robot_[i:i + shape_m[0], j:j + shape_m[1]] > 0, True, False)
        valid = not np.any(np.logical_and(module_exists, robot_exists))
        if valid:
            valid_coordinates.append(coordinate)
    return valid_coordinates

def custom_agg(series):
    return next((item for item in series if item == item), np.nan)

def convert_and_sort(key1, key2):
    key1 = int(key1)
    pair = (key1, key2) if key1 < key2 else (key2, key1)
    return pair

def make_random_robot(modules, possible_coordinates, p, min_module_num, size_limit):
    keys = list(modules.keys())
    valid_n = np.random.choice(np.arange(len(p)), p=p) + min_module_num
    

    while True:
        np.random.shuffle(keys)
        voxel_num = sum([np.sum(modules[key].module > 0) for key in keys[:valid_n]])
        actuator_num = sum([np.sum(modules[key].module > 2) for key in keys[:valid_n]])
        if actuator_num / voxel_num > 0.3:
            break
    
    robot = modules[keys[0]].module.copy()
    module_map = np.where(robot > 0, modules[keys[0]].key, 0).astype(int)    

    valid = True
    valid_modules = pd.DataFrame({"coordinate_i": [0], "coordinate_j": [0]}, columns=["coordinate_i", "coordinate_j"], index=[keys[0]])
    for key_i in keys[1:valid_n]:

        module_key = modules[key_i].key
        module = modules[key_i].module

        possibles = []
        for module_key_ in valid_modules.index:
            pair = (module_key_, module_key)
            possibles_ = possible_coordinates[pair].copy()
            possibles_[["coordinate_i", "coordinate_j"]] += valid_modules.loc[module_key_, ["coordinate_i", "coordinate_j"]].values
            possibles.append(possibles_)
        possibles = pd.concat(possibles)
        possibles = possibles.groupby(["coordinate_i", "coordinate_j"]).agg(custom_agg)
        coordinates = possibles.index.values

        coordinates = check_adjacent_validity(robot, module, coordinates, size_limit)
        if len(coordinates) == 0:
            valid = False
            break

        coordinate = coordinates[np.random.choice(len(coordinates))]

        robot, module_map, valid_modules = connect_module(robot, module_map, valid_modules, module, module_key, coordinate)

        if robot.shape[0] > size_limit[0] or robot.shape[1] > size_limit[1]:
            valid = False
            print("error: size limit")
            break
    
    return valid, valid_n, robot, module_map



def to_hash(robot):
    hash = ",".join(["".join(map(str, c)) for c in robot])
    return hash

class Robot:
    def __init__(self, key, robot, module_num, module_map):
        self.key = key
        self.robot = robot
        self.module_num = module_num
        self.module_map = module_map


def sampling_robots(sample_n, modules, possible_coordinates, p, min_module_num, max_module_num, size_limit):
    robots = {}
    errors = {n: 0 for n in range(min_module_num, max_module_num + 1)}
    valids = {n: 0 for n in range(min_module_num, max_module_num + 1)}
    invalids = {n: 0 for n in range(min_module_num, max_module_num + 1)}
    for n in range(sample_n):
        valid, module_num, robot, module_map = make_random_robot(modules, possible_coordinates, p, min_module_num, size_limit)
        
        if not valid:
            errors[module_num] += 1
            continue

        hash = to_hash(robot)
        if hash in robots:
            invalids[module_num] += 1
            continue

        robots[hash] = Robot(0, robot, module_num, module_map)
        valids[module_num] += 1
    return robots, valids, invalids, errors



def get_surface(robot):
    robot_ = np.pad(robot, 1)
    directions = np.array([[1, 0], [0, 1], [0, -1], [-1, 0]])
    positions = np.vstack([*np.where(robot_ > 0)]).T
    check_positions = np.expand_dims(positions, axis=1) + np.expand_dims(directions, axis=0)
    surface = np.sum(robot_[check_positions[:, :, 0], check_positions[:, :, 1]] == 0)
    return surface

def mapping(robot, module_num):
    shape = np.array(robot.robot.shape)
    weight = np.sum(robot.robot > 0)
    surface = get_surface(robot.robot)
    voxel_density = np.array([np.mean(robot.robot > 0)])
    density = np.array([np.sum(robot.robot == 3), np.sum(robot.robot==4)]) / weight

    geometric = np.array(robot.robot.shape) / 2
    centroids = np.array([np.mean(idx) for idx in np.where(robot.module_map > 0)])
    relative = centroids - geometric + 0.5

    valid = np.zeros(module_num)
    pos = np.full((module_num, 2), np.nan)
    
    keys = np.unique(robot.module_map[robot.module_map > 0])
    for key in keys:
        valid[key - 1] = 1
        pos[key - 1] = np.array([np.mean(idx) for idx in np.where(robot.module_map == key)]) - centroids

    features = np.hstack([shape, weight, surface, surface / weight, voxel_density, density, relative, valid, pos.flatten()])
    return features.astype("float32")




def diff_log_likelihood_of_N(N, M, U):
    return np.log(N + 1) - np.log(N - U + 1) + (np.log(N) - np.log(N + 1)) * M

def search_maximum_likelihood(M, U):
    if M == U:
        U -= 1
    if diff_log_likelihood_of_N(U, M, U) <= 0:
        return U
    
    low = U
    high = U * 2
    while diff_log_likelihood_of_N(high, M, U) > 0:
        low = high
        high *= 2
    
    while high - low > 1:
        mid = (low + high) // 2
        if diff_log_likelihood_of_N(mid, M, U) > 0:
            low = mid
        else:
            high = mid
    
    return high




from concurrent.futures import ProcessPoolExecutor, as_completed

def main():
    min_module_num = 3
    max_module_num = 8
    modules, possible_coordinates = make_stable_modules()

    make_robot_num = 10000000
    size_limit = (5, 5)

    save_path = os.path.join("robot_samples", f"{size_limit[0]}-{size_limit[1]}_{make_robot_num}")
    os.makedirs(save_path, exist_ok=True)

    count_file = os.path.join(save_path, "count.npy")
    robot_file = os.path.join(save_path, "robots.pickle")
    feature_file = os.path.join(save_path, "features.npy")

    initialize = False
    # initialize = True


    batch_n = [100, 10000]
    process_num = 10

    if initialize:
        robots = {}
        counts = np.zeros((4, max_module_num - min_module_num + 1), dtype=int)
        p = np.ones(max_module_num - min_module_num + 1, dtype=int) / (max_module_num - min_module_num + 1)
    else:
        with open(robot_file, "rb") as f:
            robots = pickle.load(f)
        counts = np.load(count_file)
        expected_N = np.array([search_maximum_likelihood(counts[0][i], counts[1][i]) for i in range(max_module_num - min_module_num + 1)]).astype(int)
        p = np.log(counts[1] / expected_N)
        p = p / np.sum(p)
    

    error_count = np.sum(counts[3])
    generate_count = np.sum(counts[0]) + error_count

    uique_count = len(robots)
    base_time = time.time()
    while uique_count < make_robot_num:
        t = time.time()
        n = min(min(batch_n[1], batch_n[0] + generate_count // process_num // 10), (make_robot_num - uique_count - 1) // process_num + 1)
        generates_iter = n * process_num
        print(f"generating {n: =,} x {process_num: =,} = {generates_iter: =,}", end="")

        args = [n, modules, possible_coordinates, p, min_module_num, max_module_num, size_limit]
        now_count = len(robots)
        with ProcessPoolExecutor(max_workers=process_num) as executor:
            futures = [executor.submit(sampling_robots, *args) for _ in range(process_num)]

            for future in as_completed(futures):
                generated, valids, invalids, errors = future.result()

                for d_hash in set(generated.keys()) & set(robots.keys()):
                    module_num = generated[d_hash].module_num
                    valids[module_num] -= 1
                    invalids[module_num] += 1

                counts += np.vstack([  # M, U, D, E
                    np.array([valids[n] + invalids[n] for n in range(min_module_num, max_module_num + 1)]),
                    np.array([valids[n] for n in range(min_module_num, max_module_num + 1)]),
                    np.array([invalids[n] for n in range(min_module_num, max_module_num + 1)]),
                    np.array([errors[n] for n in range(min_module_num, max_module_num + 1)])
                ])

                error_count += n - len(generated)
                robots.update(generated)

        expected_N = np.array([search_maximum_likelihood(counts[0][i], counts[1][i]) for i in range(max_module_num - min_module_num + 1)]).astype(int)
        p = np.log(counts[1] / expected_N)
        p = p / np.sum(p)

        generate_count += generates_iter
        uique_count = len(robots)
        duplicate_count = generate_count - uique_count - error_count

        t_ = time.time()
        spend = t_ - t
        ave = spend / n
        new_count = len(robots) - now_count
        print(f"\rgenerate: {generate_count: =8}  unique: {uique_count: =8}  duplicate: {duplicate_count: =8}  error: {error_count: =8}  ratio: {duplicate_count/uique_count: =.3f}  new: {new_count: 5}  elapsed: {spend: =.2f} sec ({ave: =.4f} s/g)  total: {(t_ - base_time) / 60: =.2f} min")
        print("\tM: [" + ", ".join(f"{c: =15,}" for c in counts[0]) + "]")
        print("\tU: [" + ", ".join(f"{c: =15,}" for c in counts[1]) + "]")
        print("\tD: [" + ", ".join(f"{c: =15,}" for c in counts[2]) + "]")
        print("\tE: [" + ", ".join(f"{c: =15,}" for c in counts[3]) + "]")
        print("\tN: [" + ", ".join(f"{n: =15,}" for n in expected_N) + "]")
        print("\tP: [" + ", ".join(f"{p_: =15.6f}" for p_ in p) + "]")
        t = t_

        with open(robot_file, "wb") as f:
            pickle.dump(robots, f)
        np.save(count_file, counts)

    features = []
    for robot in robots.values():
        features.append(mapping(robot, len(modules)))

    features = np.vstack(features)
    np.save(feature_file, features)

        

if __name__=="__main__":
    main()


