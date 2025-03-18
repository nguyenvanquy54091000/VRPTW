import numpy as np
import random
import time
from collections import namedtuple
import pandas as pd
import os

Customer = namedtuple('Customer', ['id', 'xy_coord', 'demand', 'readyTime', 'dueTime', 'serviceTime'])


def load_csv_dataset(name_of_id, number_of_customer):
    path = name_of_id 
    f = pd.read_csv(path)
    data = np.array(f[['XCOORD.', 'YCOORD.']])
    demands = np.array(f['DEMAND'])
    ready_time = np.array(f['READY TIME'])
    due_date = np.array(f['DUE DATE'])
    service_time = np.array(f['SERVICE TIME'])
    customers = []
    for i in range(number_of_customer + 1):
        customers.append(Customer(i, data[i], demands[i], ready_time[i], due_date[i], service_time[i]))
    return data[:number_of_customer + 1], customers

# Tính khoảng cách giữa hai điểm
def distance_cdist(X, Y, metric='euclidean'):
    from scipy.spatial.distance import cdist
    return cdist(X, Y, metric=metric)


class Kmeans:
    def __init__(self, epsilon=1e-5, maxiter=1000, n_cluster=3):
        self.__epsilon = epsilon
        self.__maxiter = maxiter
        self.n_cluster = n_cluster

    def __init__cluster_center(self, X, seed=0):
        if seed > 0:
            np.random.seed(seed)
        return X[np.random.choice(X.shape[0], self.n_cluster, replace=False)]

    #cập nhật vị trí của các tâm cụm dựa trên các điểm dữ liệu được gán vào tâm đó
    def update_cluster_center(self, X, label):
        return [np.mean(X[label == i], axis=0) for i in range(self.n_cluster)]

    #Cập nhật và gán mỗi điểm vào cụm có vị trí gần tâm nhất
    def update_membership_matrix(self, X, V):
        return np.argmin(distance_cdist(X, V), axis=1)

    #kiểm tra tính hội tụ
    def has_converged(self, centers, new_centers):
        return set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers])

    def k_means(self, X, seed=42):
        v = self.__init__cluster_center(X, seed)
        for step in range(self.__maxiter):
            old_v = v.copy()
            u = self.update_membership_matrix(X, old_v)
            v = self.update_cluster_center(X, u)
            if self.has_converged(old_v, v):
                break
        return u, v, step + 1

    def data_to_cluster(self, U):
        # return [np.argwhere(U == i).T[1] + 1 for i in range(self.n_cluster)]
        # return [np.argwhere(U == i).flatten() + 1 for i in range(self.n_cluster)]
        return [np.argwhere(U == i).T[0] + 1 for i in range(self.n_cluster)]


# Solomon Insertion Heuristic
def solomon_insertion_heuristic(customers, capacity):
    unrouted = list(range(1, len(customers)))  # Bỏ depot
    route = [0]  # Bắt đầu từ depot
    current_capacity = 0
    current_time = 0

    while unrouted:
        best_customer = None
        best_position = None
        best_cost = float('inf')

        for customer_id in unrouted:
            customer = customers[customer_id]
            for pos in range(1, len(route) + 1):
                prev_node = customers[route[pos - 1]]
                next_node = customers[route[pos] if pos < len(route) else 0]
                dist_to_customer = np.linalg.norm(prev_node.xy_coord - customer.xy_coord)
                dist_from_customer = np.linalg.norm(customer.xy_coord - next_node.xy_coord)
                arrival_time = current_time + dist_to_customer
                if arrival_time < customer.readyTime:
                    arrival_time = customer.readyTime
                if arrival_time + customer.serviceTime + dist_from_customer > next_node.dueTime:
                    continue
                if current_capacity + customer.demand > capacity:
                    continue
                cost = dist_to_customer + dist_from_customer - np.linalg.norm(prev_node.xy_coord - next_node.xy_coord)
                if cost < best_cost:
                    best_cost = cost
                    best_customer = customer_id
                    best_position = pos
        #tìm thấy khách hàng tốt nhất để chèn vào tuyến
        if best_customer:
            route.insert(best_position, best_customer)
            current_capacity += customers[best_customer].demand
            unrouted.remove(best_customer)
        else:
            break #không thể chèn thêm, kết thúc tuyến
    return route + [0]  #quay về kho

# Lớp Individual
class Individual:
    def __init__(self, customerList=None, fitness=0, distance=0):
        self.customerList = customerList
        self.fitness = fitness
        self.distance = distance

# Lớp GA
class GA:
    def __init__(self, individual=4500, generation=100, crossover_rate=0.8, mutation_rate=0.15, vehicle_capacity=200, conserve_rate=0.1, M=50, customers=None):
        self._individual = individual
        self._generation = generation
        self._crossover_rate = crossover_rate
        self._mutation_rate = mutation_rate
        self._vehicle_capacity = vehicle_capacity
        self._conserve_rate = conserve_rate
        self._M = M
        self.customers = customers
        self.best_fitness_global = 0
        self.route_count_global = 0
        self.best_distance_global = 0
        self.best_route_global = []
        self.process_time = 0


 
    def initialPopulation(self, cluster, use_heuristic=False): #hybrid_init=False
        if use_heuristic:
            #khởi tạo 1 cá thể từ SIH
            route = solomon_insertion_heuristic(self.customers, self._vehicle_capacity)
            self.__population = [Individual(customerList=route)]
            #thêm các cá thể ngẫu nhiên từ cụm cho đến khi đủ self._individual
            for _ in range(self._individual - 1):
                random_route = np.random.permutation(cluster)
                self.__population.append(Individual(customerList=random_route))
        else:
            self.__population = [Individual(customerList=np.random.permutation(cluster)) for _ in range(self._individual)]

        #Khởi tạo một nửa quần thể ngẫu nhiên và một nửa bằng heuristic
        # if hybrid_init:
        #     half_individual = self._individual // 2
        #     # Phần heuristic
        #     heuristic_route = solomon_insertion_heuristic(self.customers, self._vehicle_capacity)
        #     self.__population = [Individual(customerList=heuristic_route) for _ in range(half_individual)]
        #     # Phần ngẫu nhiên
        #     for _ in range(half_individual, self._individual):
        #         random_route = np.random.permutation(cluster)
        #         self.__population.append(Individual(customerList=random_route))
        # elif use_heuristic:
        #     # Khởi tạo toàn bộ bằng heuristic
        #     route = solomon_insertion_heuristic(self.customers, self._vehicle_capacity)
        #     self.__population = [Individual(customerList=route) for _ in range(self._individual)]
        # else:
        #     # Khởi tạo toàn bộ ngẫu nhiên
        #     self.__population = [Individual(customerList=np.random.permutation(cluster)) for _ in range(self._individual)]


    def individualToRoute(self, individual):
        individual = np.array(individual).astype(int)
        route = []
        sub_route = []
        current_capacity = 0
        current_time = 0
        last_customer_id = 0
        for customer_id in individual:
            customer = self.customers[customer_id]
            demand = customer.demand
            service_time = customer.serviceTime
            #tính thời gian di chuyển từ khách hàng trước đó đến khách hàng hiện tại
            moving_time = np.linalg.norm(customer.xy_coord - self.customers[last_customer_id].xy_coord)
            #thời gian của xe tại khách hàng hiện tại
            arrival_time = current_time + moving_time
            #tính thời gian chờ nếu đến sớm
            waiting_time = max(customer.readyTime - self._M - arrival_time, 0)
            #thời gian của khách hiện tại về kho
            return_time = np.linalg.norm(customer.xy_coord - self.customers[0].xy_coord)
            #tính thời gian hoàn thành tuyến
            update_elapsed_time = arrival_time + service_time + waiting_time + return_time
            #kểm tra rằng buộc về khối lượng và thời gian
            if (current_capacity + demand <= self._vehicle_capacity) and (update_elapsed_time <= self.customers[0].dueTime + self._M):
                sub_route.append(customer_id)
                current_capacity += demand
                current_time = arrival_time + service_time + waiting_time
            else:
                route.append(sub_route)
                sub_route = [customer_id]
                current_capacity = demand
                current_time = max(customer.readyTime, moving_time) + service_time
            last_customer_id = customer_id
        if sub_route:
            route.append(sub_route)
        return [arr for arr in route if len(arr) > 0]

    #fitness của mỗi cá thể
    def cal_fitness_individual(self, individual):
        route = self.individualToRoute(individual)
        fitness = 0
        distance = 0
        for sub_route in route:
            sub_route_distance = 0
            elapsed_time = 0
            last_customer_id = 0
            for customer_id in sub_route:
                customer = self.customers[customer_id]
                moving_time = np.linalg.norm(customer.xy_coord - self.customers[last_customer_id].xy_coord)
                sub_route_distance += moving_time
                arrival_time = elapsed_time + moving_time
                waiting_time = max(customer.readyTime - self._M - arrival_time, 0)
                #tính thời gian trễ
                delay_time = max(arrival_time - customer.dueTime - self._M, 0)
                #cộng vào giá trị fitness
                fitness += waiting_time + delay_time
                #cap nhật thời gian trôi qua
                elapsed_time = arrival_time + customer.serviceTime + waiting_time
                last_customer_id = customer_id
            return_time = np.linalg.norm(self.customers[last_customer_id].xy_coord - self.customers[0].xy_coord)
            sub_route_distance += return_time
            distance += sub_route_distance
        fitness += distance
        return fitness, distance


    def selection(self):
        #sắp xếp quần thể theo thứ tự tăng dần của fitness
        self.__population.sort(key=lambda x: x.fitness)
        # positionToDel = int(self._individual * (1 + self._conserve_rate) / 2)
        positionToDel = int(self._individual * (1 - self._conserve_rate))  # Giữ lại nhiều cá thể hơn
        if positionToDel < len(self.__population):
            del self.__population[positionToDel:]
        

    def Swap_Crossover(self, dad, mom):
        probabilities = [0.25, 0.25, 0.25, 0.25]
        choice = np.random.choice([i for i in range(len(probabilities))], p=probabilities)
        #lai ghép 1 điểm
        if choice == 0:
            pos = random.randrange(len(dad))
            filter_dad = np.setdiff1d(dad, mom[pos:], assume_unique=True)
            filter_mom = np.setdiff1d(mom, dad[pos:], assume_unique=True)
            gene_child_1 = np.hstack((filter_dad[:pos], mom[pos:]))
            gene_child_2 = np.hstack((filter_mom[:pos], dad[pos:]))
        #lai ghép 2 điểm
        elif choice == 1:
            pos1, pos2 = sorted(random.sample(range(len(dad)), 2))
            filter_dad = np.setdiff1d(dad, mom[pos1:pos2], assume_unique=True)
            filter_mom = np.setdiff1d(mom, dad[pos1:pos2], assume_unique=True)
            gene_child_1 = np.hstack((filter_dad[:pos1], mom[pos1:pos2], filter_dad[pos1:]))
            gene_child_2 = np.hstack((filter_mom[:pos1], dad[pos1:pos2], filter_mom[pos1:]))
        #lai ghép dựa trên vị trí bằng chọn 2 vị trí ngãu nhiên
        elif choice == 2:
            # Sử dụng độ dài chung của cả hai cá thể để chọn điểm cắt
            common_length = min(len(dad), len(mom))
            pos1, pos2 = sorted(random.sample(range(common_length), 2))
            # Tạo mảng con với độ dài = common_length
            gene_child_1 = np.zeros(common_length, dtype=int)
            gene_child_2 = np.zeros(common_length, dtype=int)
            mapping_gene = np.vstack((dad[pos1:pos2], mom[pos1:pos2]))
            # for idx_p, (d, m) in enumerate(zip(dad, mom)):
            #     if idx_p in range(pos1, pos2):
            #         gene_child_1[idx_p] = mom[idx_p]
            #         gene_child_2[idx_p] = dad[idx_p]
            #         continue
            #     i, idx = 1, idx_p
            for idx_p in range(common_length):
                if pos1 <= idx_p < pos2:
                    gene_child_1[idx_p] = mom[idx_p]
                    gene_child_2[idx_p] = dad[idx_p]
                else:
                    d = dad[idx_p]
                    m = mom[idx_p]
                    i = 1
                    # Cập nhật giá trị d dựa trên mapping_gene
                    while d in mapping_gene[i]:
                        idx_val = np.argwhere(mapping_gene[i] == d)[0][0]
                        d = mapping_gene[abs(i-1)][idx_val]
                    i = 0
                    # Cập nhật giá trị m dựa trên mapping_gene
                    while m in mapping_gene[i]:
                        idx_val = np.argwhere(mapping_gene[i] == m)[0][0]
                        m = mapping_gene[abs(i-1)][idx_val]
                    gene_child_1[idx_p] = d
                    gene_child_2[idx_p] = m

        else:
            #lai ghép tuyến đường
            route_dad = self.individualToRoute(dad)
            route_mom = self.individualToRoute(mom)

            #lấy 1 khách hàng ngẫu nhiên từ mỗi tuyến dad, mom
            sub_route_mom = route_mom[np.random.randint(0, len(route_mom))]
            sub_route_dad = route_dad[np.random.randint(0, len(route_dad))]
            #loại bỏ khách tại vị trí đó
            gene_child_1 = np.setdiff1d(dad, sub_route_mom)
            gene_child_2 = np.setdiff1d(mom, sub_route_dad)

            #chèn khách vào vị trí tốt nhất trong tuyến
            for gen in sub_route_mom:
                parts = [self.cal_fitness_individual(np.insert(gene_child_1, idx, gen))[0] for idx in range(len(gene_child_1)+1)]
                gene_child_1 = np.insert(gene_child_1, np.argmin(parts), gen)
            for gen in sub_route_dad:
                parts = [self.cal_fitness_individual(np.insert(gene_child_2, idx, gen))[0] for idx in range(len(gene_child_2)+1)]
                gene_child_2 = np.insert(gene_child_2, np.argmin(parts), gen)
        return gene_child_1, gene_child_2



    def PMX_Crossover(self, dad, mom):
        length = len(dad)
        pos1, pos2 = sorted(random.sample(range(length), 2))

        # Khởi tạo con cái với giá trị mặc định (-1 để đánh dấu chưa điền)
        child1 = np.full(length, -1, dtype=int)
        child2 = np.full(length, -1, dtype=int)
        child1[pos1:pos2] = dad[pos1:pos2]
        child2[pos1:pos2] = mom[pos1:pos2]
        #tạo ánh xạ giữa các giá trị trong đoạn đã sao chép
        mapping_dad_to_mom = {dad[i]: mom[i] for i in range(pos1, pos2)}
        mapping_mom_to_dad = {mom[i]: dad[i] for i in range(pos1, pos2)}
        # Hàm phụ để điền các phần còn lại của con cái
        def fill_child(child, parent, mapping):
            for i in range(length):
                if child[i] == -1:  # Nếu vị trí chưa được điền
                    value = parent[i]
                    # Kiểm tra và thay thế nếu giá trị đã tồn tại trong đoạn sao chép
                    while value in mapping:
                        value = mapping[value]
                    child[i] = value
        fill_child(child1, mom, mapping_mom_to_dad)
        fill_child(child2, dad, mapping_dad_to_mom)
        return child1, child2


    
    #phương pháp đột biến đảo đoạn
    def mutation(self, child):
        child_new = np.copy(child)
        pos1, pos2 = sorted(random.sample(range(len(child)), 2))
        child_new[pos1:pos2] = child_new[pos1:pos2][::-1]
        return child_new
    
    #phương pháp đột biến Swap 
    def swap_mutation(self, individual):
        individual_new = np.copy(individual)
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual_new[idx1], individual_new[idx2] = individual_new[idx2], individual_new[idx1]
        return individual_new

    def hybird(self):
            # Giũ lại các cá thể tốt nhất
            index = int(self._conserve_rate * self._individual)
            # Nếu số cá thể từ index trở đi nhỏ hơn 2, sử dụng toàn bộ population để lấy mẫu
            sample_pool = self.__population[index:] if len(self.__population[index:]) >= 2 else self.__population
            while len(self.__population) < self._individual:
                print(f"Kích thước của population: {len(self.__population)}")
                print(f"Giá trị của index: {index}")
                print(f"Kích thước của population[index:]: {len(self.__population[index:])}")
                #     dad, mom = random.sample(self.__population[index:], 2)
                dad, mom = random.sample(sample_pool, 2)
                if random.random() <= self._crossover_rate:
                    gene_child_1, gene_child_2 = self.Swap_Crossover(dad.customerList, mom.customerList)
                    # gene_child_1, gene_child_2 = self.PMX_Crossover(dad.customerList, mom.customerList)

                    if random.random() <= self._mutation_rate:
                        gene_child_1 = self.mutation(gene_child_1)
                        gene_child_2 = self.mutation(gene_child_2)
                    self.__population.append(Individual(customerList=gene_child_1))
                    if len(self.__population) < self._individual:
                        self.__population.append(Individual(customerList=gene_child_2))
                

    #chạy GA cho mỗi cụm khách hàng
    def fit(self, cluster, use_heuristic=False): #hybrid_init=False
        _start_time = time.time()
        self.initialPopulation(cluster, use_heuristic) #hybrid_init
        for _ in range(self._generation):
            #tính toán fitness cho từng cá thể trong quần thể
            for ind in self.__population:
                ind.fitness, ind.distance = self.cal_fitness_individual(ind.customerList)
            self.selection()
            self.hybird()
        self.process_time += time.time() - _start_time
        #tính toán fitness cuối cùng và chọn lọc để lấy ra cá thể tốt nhất
        for ind in self.__population:
            ind.fitness, ind.distance = self.cal_fitness_individual(ind.customerList)
        self.selection()
        #cập nhật thông tin về giải pháp tốt nhất
        self.best_fitness_global += self.__population[0].fitness
        self.best_distance_global += self.__population[0].distance
        best_route = self.individualToRoute(self.__population[0].customerList)
        self.best_route_global.append(best_route)
        self.route_count_global += len(best_route)


    def fit_allClusters(self, clusters, use_heuristic=False): #hybrid_init=False
        for cluster in clusters:
            self.fit(cluster, use_heuristic) #hybrid_init
        return self.best_fitness_global, self.best_route_global, self.best_distance_global, self.route_count_global, self.process_time


if __name__ == "__main__":
    print("Working Directory:", os.getcwd())
    N_CLUSTER = 5
    NUMBER_OF_CUSTOMER = 100
    INDIVIDUAL = 500
    GENERATION = 100
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.15
    VEHICLE_CAPACITY = 200
    CONSERVE_RATE = 0.1
    M = 50
    # DATA_ID = 'R201.csv'
    DATA_ID = r"E:\Innter\VRPTW\R201.csv"
    # url_data = ""
    # data = pd.read_csv('R201.csv')

    # data, customers = load_csv_dataset(url_data, DATA_ID, NUMBER_OF_CUSTOMER)
    data, customers = load_csv_dataset(DATA_ID, NUMBER_OF_CUSTOMER)
    kmeans = Kmeans(n_cluster=N_CLUSTER)
    data_kmeans = np.delete(data, 0, 0)
    U, V, step = kmeans.k_means(data_kmeans)
    clusters = kmeans.data_to_cluster(U)

    # GA khởi tạo quần thể ngẫu nhiên
    ga_random = GA(individual=INDIVIDUAL, generation=GENERATION, crossover_rate=CROSSOVER_RATE, mutation_rate=MUTATION_RATE,
                   vehicle_capacity=VEHICLE_CAPACITY, conserve_rate=CONSERVE_RATE, M=M, customers=customers)
    best_fitness_random, best_route_random, best_distance_random, route_count_random, process_time_random = ga_random.fit_allClusters(clusters, use_heuristic=False)

    # GA khởi tạo bằng Solomon Insertion Heuristic
    ga_heuristic = GA(individual=INDIVIDUAL, generation=GENERATION, crossover_rate=CROSSOVER_RATE, mutation_rate=MUTATION_RATE,
                      vehicle_capacity=VEHICLE_CAPACITY, conserve_rate=CONSERVE_RATE, M=M, customers=customers)
    best_fitness_heuristic, best_route_heuristic, best_distance_heuristic, route_count_heuristic, process_time_heuristic = ga_heuristic.fit_allClusters(clusters, use_heuristic=True)

    # # GA khởi tạo kết hợp SIH + random
    # ga_hybrid = GA(individual=INDIVIDUAL, generation=GENERATION, crossover_rate=CROSSOVER_RATE, mutation_rate=MUTATION_RATE,
    #                vehicle_capacity=VEHICLE_CAPACITY, conserve_rate=CONSERVE_RATE, M=M, customers=customers)
    # best_fitness_hybrid, best_route_hybrid, best_distance_hybrid, route_count_hybrid, process_time_hybrid = ga_hybrid.fit_allClusters(clusters, use_heuristic=False, hybrid_init=True)

  
    print("Kết quả với khởi tạo ngẫu nhiên:")
    print(f"Fitness: {best_fitness_random:.2f}")
    print(f"Số lượng xe: {route_count_random}")
    print(f"Thời gian chạy: {process_time_random:.2f} giây")

    print("\nKết quả với khởi tạo bằng Solomon Insertion Heuristic:")
    print(f"Fitness: {best_fitness_heuristic:.2f}")
    print(f"Số lượng xe: {route_count_heuristic}")
    print(f"Thời gian chạy: {process_time_heuristic:.2f} giây")

    # print("\nKết quả với khởi tạo kết hợp (một nửa ngẫu nhiên, một nửa heuristic):")
    # print(f"Fitness: {best_fitness_hybrid:.2f}")
    # print(f"Số lượng xe: {route_count_hybrid}")
    # print(f"Thời gian chạy: {process_time_hybrid:.2f} giây")