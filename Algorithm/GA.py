import numpy as np
import pandas as pd
import random
import math
import time
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


# Trao đổi chéo thứ tự
def order_crossover(parent1,parent2):
    size = len(parent1)
    child = [None] * size
    a, b = sorted(random.sample(range(size), 2))
    child[a:b+1] = parent1[a:b+1]
    pos = (b + 1) % size
    for gene in parent2[b+1:] + parent2[:b+1]:
        if gene not in child:
            child[pos] = gene
            pos = (pos + 1) % size
    return child

# Đột biến hoán đổi
def swap_mutation(chromosome):
    if len(chromosome) >= 2:
        a, b = random.sample(range(len(chromosome)), 2)
        chromosome[a], chromosome[b] = chromosome[b], chromosome[a]
    return chromosome



# Giải mã cá thế theo số ngiueen
def decode_chromosome(chromosome, customers, capacity=200, penalty_factor=10):
    routes = []
    depot_id = customers[0]['CUST NO.']
    route = [depot_id]
    current_load = 0
    current_time = 0
    total_penalty = 0
    cust_dict = {cust['CUST NO.']: cust for cust in customers}
    
    for cust_id in chromosome:
        cust = cust_dict[cust_id]
        if current_load + cust['DEMAND']> capacity:
            route.append(depot_id)
            routes.append(route)
            route = [depot_id]
            current_load= 0
            current_time = 0
        

        last_cust = cust_dict[route[-1]]
        travel_time = math.hypot(cust['XCOORD.'] - last_cust['XCOORD.'], cust['YCOORD.'] - last_cust['YCOORD.'])
        arrival_time = current_time + travel_time
        
        if arrival_time < cust['READY TIME']:
            arrival_time = cust['READY TIME']
        if arrival_time > cust['DUE DATE']:
            total_penalty += penalty_factor * (arrival_time - cust['DUE DATE'])
        
        current_time = arrival_time + cust['SERVICE TIME']
        current_load += cust['DEMAND']
        route.append(cust_id)
    
    route.append(depot_id)
    routes.append(route)
    return routes, total_penalty

# Tổng chi phí của một cá thể
def compute_cost(chromosome, customers, capacity=200, penalty_factor=10):
    routes, penalty = decode_chromosome(chromosome, customers, capacity, penalty_factor)
    total_distance = 0
    cust_dict = {cust['CUST NO.']: cust for cust in customers}
    for route in routes:
        for i in range(len(route) - 1):
            c1 = cust_dict[route[i]]
            c2 = cust_dict[route[i+1]]
            total_distance += math.hypot(c2['XCOORD.'] - c1['XCOORD.'], c2['YCOORD.'] - c1['YCOORD.'])
    return total_distance + penalty

# Tính giá trị độ thích nghi của mỗi cá thể
def fitness(chromosome, customers, capacity=200, penalty_factor=10):
    cost = compute_cost(chromosome, customers, capacity, penalty_factor)
    return 1.0 / (cost + 1e-6)

# Tính toán trả tổng quãng đường, tổng phạt, các lộ trình đã tạo

def get_route_details(chromosome, customers, capacity=200, penalty_factor=10):
    routes, total_penalty = decode_chromosome(chromosome, customers, capacity, penalty_factor)
    total_distance = 0
    cust_dict = {cust['CUST NO.']: cust for cust in customers}
    for route in routes:
        for i in range(len(route) - 1):
            c1 = cust_dict[route[i]]
            c2 = cust_dict[route[i+1]]
            total_distance += math.hypot(c2['XCOORD.'] - c1['XCOORD.'], c2['YCOORD.'] - c1['YCOORD.'])
    return total_distance, total_penalty, routes

# Tìm số cụm tối ưu
# def find_optimal_clusters(data, max_k):
#     iters = range(2, max_k + 1)
#     sse = []
#     silhouette_scores = []

#     for k in iters:
#         kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
#         sse.append(kmeans.inertia_)
#         silhouette_scores.append(silhouette_score(data, kmeans.labels_))

#     f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
#     ax1.plot(iters, sse, marker='o')
#     ax1.set_xlabel('Cluster Centers')
#     ax1.set_ylabel('SSE')
#     ax1.set_title('Elbow Method')

#     ax2.plot(iters, silhouette_scores, marker='o')
#     ax2.set_xlabel('Cluster Centers')
#     ax2.set_ylabel('Silhouette Score')
#     ax2.set_title('Silhouette Score Method')

#     plt.show()

#     # Determine the optimal number of clusters based on the Elbow Method and Silhouette Score
#     optimal_k = iters[silhouette_scores.index(max(silhouette_scores))]
#     return optimal_k


# GA cho 1 cụm
def GA_for_cluster(customers, pop_size=150, generations=500, 
                   crossover_rate=0.8, mutation_rate=0.15, elitism_rate=0.1, 
                   capacity=200, penalty_factor=12):
    depot_id = customers[0]['CUST NO.']
    customer_ids = [cust['CUST NO.'] for cust in customers if cust['CUST NO.'] != depot_id]
    if not customer_ids:
        return [], 0
    population = []
    for _ in range(pop_size):
        perm = customer_ids.copy()
        random.shuffle(perm)
        population.append(perm)
    
    best_cost = float('inf')
    best_solution = None
    
    for gen in range(generations):
        fitness_values = [fitness(ind, customers, capacity, penalty_factor) for ind in population]
        costs = [compute_cost(ind, customers, capacity, penalty_factor) for ind in population]
        
        elite_count = max(1, int(elitism_rate * pop_size))
        elite_indices = np.argsort(costs)[:elite_count]
        new_population = [population[i] for i in elite_indices]
        while len(new_population) < pop_size:
            total_fit = sum(fitness_values)
            if total_fit == 0:
                new_population.extend(random.choices(population, k=pop_size - len(new_population)))
                break
            parent1, parent2 = random.choices(population, weights=fitness_values, k=2)
            if random.random() < crossover_rate and len(parent1) >= 2:
                child = order_crossover(parent1, parent2)
            else:
                child = parent1.copy()
            if random.random() < mutation_rate:
                child = swap_mutation(child)
            new_population.append(child)
        
        population = new_population
        current_best_idx = np.argmin(costs)
        current_best_cost = costs[current_best_idx]
        if current_best_cost < best_cost:
            best_cost = current_best_cost
            best_solution = population[current_best_idx]
        if gen % 100 == 0:
            print(f"Generation {gen}: Best Cost = {best_cost:.2f}")
    
    return best_solution, best_cost


# Chạy GA không phân cụm cho toàn bộ dữ liệu
def run_GA_entire(customers_all, pop_size=150, generations=500, 
                  crossover_rate=0.8, mutation_rate=0.15, elitism_rate=0.1, 
                  capacity=200, penalty_factor=12):
    print("Chạy GA đơn giản ...")
    start = time.time()
    best_solution, best_cost = GA_for_cluster(customers_all, pop_size, generations, 
                                              crossover_rate, mutation_rate, elitism_rate, 
                                              capacity, penalty_factor)
    end = time.time()
    total_distance, total_penalty, routes = get_route_details(best_solution, customers_all, capacity, penalty_factor)
    runtime = end - start
    print("Các tuyến đường GA:", routes)
    print("Tổng chi phí GA:", best_cost)
    print("Tổng quãng đường GA:", total_distance)
    print("Tổng Phạt GA:", total_penalty)
    print(f"Thời gian chạy GA: {runtime:.2f} giây")
    return best_cost, total_distance, runtime

# Chạy GA với kmean
def run_GA_with_clustering(customers_all, k, init_method='random', pop_size=150, generations=1000, 
                           crossover_rate=0.8, mutation_rate=0.15, elitism_rate=0.1, 
                           capacity=200, penalty_factor=12):
    print(f"\nChạy GA+Kmean sủw dụng init='{init_method}'...")
    start = time.time()
    depot = customers_all[0]
    customers = customers_all[1:]
    customer_coords = np.array([[c['XCOORD.'], c['YCOORD.']] for c in customers])
    
    kmeans = KMeans(n_clusters=k, random_state=42, init=init_method).fit(customer_coords)
    labels = kmeans.labels_
    
    cluster_customers = {i: [] for i in range(k)}
    for idx, cust in enumerate(customers):
        cluster_label = labels[idx]
        cluster_customers[cluster_label].append(cust)
    
    total_cost = 0
    total_distance_all = 0
    run_time_cluster = 0
    final_routes = {}
    for cluster_label, cust_list in cluster_customers.items():
        if not cust_list:
            continue
        
        print(f"\nXử lý cụm {cluster_label} với {len(cust_list)} khách hàng...")
        cluster_data = [depot] + cust_list
        sol_start = time.time()
        best_solution, best_cost = GA_for_cluster(
            cluster_data,
            pop_size=pop_size,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elitism_rate=elitism_rate,
            capacity=capacity,
            penalty_factor=penalty_factor
        )
        sol_end = time.time()
        distance, penalty, routes = get_route_details(best_solution, cluster_data, capacity, penalty_factor)
        final_routes[cluster_label] = routes
        total_cost += best_cost
        total_distance_all += distance
        cluster_runtime = sol_end - sol_start
        run_time_cluster += cluster_runtime
        print(f"chi phí cụm {cluster_label}: {best_cost:.2f}")
        print("Tuyến đường:", routes)
        print(f"Tổng quãng đường cụm {cluster_label} :{distance:.2f}")
        print(f"Thời gian chạy cụm {cluster_label}:{cluster_runtime:.2f} giây")
    
    end = time.time()
    total_runtime = end - start
    print(f"\nTổng chi phí (init='{init_method}'): {total_cost:.2f}")
    print(f"Tổng quãng đường (init='{init_method}'):{total_distance_all:.2f}")
    print(f"Tổng thời gian chạy phân cụm với tổng số cụm: {run_time_cluster:.2f} seconds")
    print(f"Tổng GA + thời gian chạy phân cụm với tổng số cụm: {total_runtime:.2f} seconds")
    return total_cost, total_distance_all, total_runtime

def main():
    overall_start = time.time()
    try:
        data = pd.read_csv("../Data/R201.csv")
        print(data.head())
    except Exception as e:
        print("Không thấy file dữ liệu.")
        return

    customers_all = data.to_dict('records')
    depot = customers_all[0]
    print("Depot ID:", depot['CUST NO.'])
    
    # GA
    cost_plain, distance_plain, runtime_plain = run_GA_entire(customers_all, pop_size=150, generations=500, 
                                                               crossover_rate=0.8, mutation_rate=0.15, elitism_rate=0.1, 
                                                               capacity=200, penalty_factor=12)
    
    # k = find_optimal_clusters(data, 15)
    k = 10
    # GA+KMeans phương thức random
    cost_kmeans_random, distance_kmeans_random, runtime_kmeans_random = run_GA_with_clustering(customers_all, k=k, init_method='random', 
                                                                                                pop_size=150, generations=500, 
                                                                                                crossover_rate=0.8, mutation_rate=0.15, elitism_rate=0.1, 
                                                                                                capacity=200, penalty_factor=12)
    
    # GA+KMeans++
    cost_kmeans_pp, distance_kmeans_pp, runtime_kmeans_pp = run_GA_with_clustering(customers_all, k=k, init_method='k-means++', 
                                                                                    pop_size=150, generations=500, 
                                                                                    crossover_rate=0.8, mutation_rate=0.15, elitism_rate=0.1, 
                                                                                    capacity=200, penalty_factor=12)
    
    print("\n=== So sánh ===")
    print(f"Tổng chi phí GA: {cost_plain:.2f}, Tổng quãng đường: {distance_plain:.2f}, Tổng thời gian: {runtime_plain:.2f} giây")
    print(f"Tổng chi phí GA + Kmean: {cost_kmeans_random:.2f}, Tổng quãng đường: {distance_kmeans_random:.2f}, Tổng thời gian: {runtime_kmeans_random:.2f} giây")
    print(f"Tổng chi phí GA + Kmean++: {cost_kmeans_pp:.2f}, Tổng quãng đường: {distance_kmeans_pp:.2f}, Tổng thời gian: {runtime_kmeans_pp:.2f} giây")
    
    methods = ['GA', 'GA+KMeans', 'GA+KMeans++']
    costs = [cost_plain, cost_kmeans_random, cost_kmeans_pp]
    distances = [distance_plain, distance_kmeans_random, distance_kmeans_pp]
    runtimes = [runtime_plain, runtime_kmeans_random, runtime_kmeans_pp]
    
    plt.figure(figsize=(12,8))
    plt.subplot(2,1,1)
    plt.bar(methods, costs, color=['blue', 'green', 'orange'])
    plt.ylabel('Tổng chi phhis')
    plt.title('So sánh chi phí các phương pháp')
    
    plt.subplot(2,1,2)
    plt.bar(methods, distances, color=['blue', 'green', 'orange'])
    plt.ylabel('Run Time (s)')
    plt.title('So sánh thời gian chạy các phương pháp')
    plt.tight_layout()
    plt.show()

    plt.subplot(2,1,3)
    plt.bar(methods, runtimes, color=['blue', 'green', 'orange'])
    plt.ylabel('Run Time (s)')
    plt.title('So sánh thời gian chạy các phương pháp')
    plt.tight_layout()
    plt.show()
    
    overall_end = time.time()
    total_run_time = overall_end - overall_start
    print(f"\nTổng thời gian chạy thuật toán: {total_run_time:.2f} giây")

if __name__ == "__main__":
    main()
