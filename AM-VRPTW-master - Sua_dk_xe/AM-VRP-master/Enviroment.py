import tensorflow as tf

class VRPproblem():
    VEHICLE_CAPACITY = 1.0  # Chuẩn hóa nhu cầu trọng tải
    ALPHA_PENALTY = 0.5
    BETA_PENALTY = 1.0
    SPEED = 1.0
    MAX_VEHICLES_K = 5 # Số xe tối đa cho một instance

    # Các hằng số này có thể được truyền vào __init__ hoặc lấy từ config
    def __init__(self, input_data, max_vehicles_k_config=None):
        depot_coords_batch = input_data[0]
        depot_time_windows_batch = input_data[1]
        customer_coords_batch = input_data[2]
        customer_demands_batch = tf.cast(input_data[3], tf.float32)
        customer_ready_times_batch = tf.cast(input_data[4], tf.float32)
        customer_due_dates_batch = tf.cast(input_data[5], tf.float32)
        customer_service_times_batch = tf.cast(input_data[6], tf.float32)

        self.batch_size, self.n_loc, _ = customer_coords_batch.shape

        self.coords = tf.concat((depot_coords_batch[:, None, :], customer_coords_batch), axis=1)

        depot_ready_time_val = depot_time_windows_batch[:, 0:1]
        depot_due_date_val = depot_time_windows_batch[:, 1:2]
        depot_service_time_val = depot_time_windows_batch[:, 2:3]
        depot_demand_val = tf.zeros((self.batch_size, 1), dtype=tf.float32)

        self.demand = tf.concat([depot_demand_val, customer_demands_batch], axis=1)
        self.ready_time = tf.concat([depot_ready_time_val, customer_ready_times_batch], axis=1)
        self.due_date = tf.concat([depot_due_date_val, customer_due_dates_batch], axis=1)
        self.service_time = tf.concat([depot_service_time_val, customer_service_times_batch], axis=1)

        self.ids = tf.range(self.batch_size, dtype=tf.int64)[:, None]

        # Trạng thái của xe HIỆN TẠI đang được điều phối trong mỗi instance của batch
        self.prev_a = tf.zeros((self.batch_size, 1), dtype=tf.int32) # Nút trước đó (0 là depot)
        self.used_capacity = tf.zeros((self.batch_size, 1), dtype=tf.float32)
        self.current_time = tf.zeros((self.batch_size, 1), dtype=tf.float32) # Thời gian hiện tại của xe

        # Trạng thái chung: khách hàng nào đã được thăm bởi BẤT KỲ xe nào
        # Shape: (batch_size, 1, self.n_loc) - chỉ theo dõi khách hàng (không phải depot)
        self.visited_customers_overall = tf.zeros((self.batch_size, 1, self.n_loc), dtype=tf.uint8)

        # Tổng số hành động (bước chọn nút) đã thực hiện trong decoder
        self.i = tf.zeros(1, dtype=tf.int64) # Hoặc tf.zeros([], dtype=tf.int64)

        # Quản lý số lượng xe đã sử dụng cho mỗi instance
        if max_vehicles_k_config is not None:
            self.MAX_VEHICLES_K = max_vehicles_k_config
        # Đếm số xe đã bắt đầu một tour (khởi tạo là 1 cho xe đầu tiên)
        self.vehicles_dispatched_count = tf.ones((self.batch_size, 1), dtype=tf.int32)

    
    def _get_travel_time(self, coords_from, coords_to):
        """
        Tính thời gian di chuyển giữa hai nút dựa trên khoảng cách Euclidean 
        và thời gian di chuyển trên mỗi đơn vị khoảng cách (tốc đọ di chuyển của xe)
        """
        return tf.norm(coords_to - coords_from, axis=-1) / self.SPEED

    def all_customers_visited_per_instance(self):
        """Kiểm tra xem tất cả khách hàng đã được thăm cho mỗi instance trong batch chưa."""
        # self.visited_customers_overall shape: (batch_size, 1, self.n_loc)
        return tf.reduce_all(tf.cast(self.visited_customers_overall, tf.bool), axis=[1, 2]) # (batch_size,)

    def all_finished(self):
        """
        Trả về True nếu TẤT CẢ các instance trong batch đã hoàn thành
        (ví dụ: tất cả khách hàng đã được thăm).
        Vòng lặp while trong GraphAttentionDecoder mong đợi một boolean duy nhất.
        """
        return tf.reduce_all(self.all_customers_visited_per_instance())

    def get_mask(self):
        """
        Trả về mask (batch_size, 1, n_nodes) cho các hành động (nút) có thể chọn.
        True trong mask nghĩa là nút đó BỊ CHE (không thể chọn).
        """
        batch_size = self.batch_size
        n_total_nodes = self.n_loc + 1

        # --- Mask cho Khách hàng (nodes 1 đến n_loc) ---
        # 1. Khách hàng đã được thăm bởi bất kỳ xe nào
        mask_visited_customers = tf.cast(self.visited_customers_overall, tf.bool)

        # 2. Ràng buộc về sức chứa của xe HIỆN TẠI
        customer_demands = self.demand[:, 1:]
        exceeds_capacity = (customer_demands + self.used_capacity) > self.VEHICLE_CAPACITY
        mask_exceeds_capacity = exceeds_capacity[:, tf.newaxis, :]

        # 3. Ràng buộc về Time Window
        current_node_coords = tf.gather_nd(self.coords, tf.concat([self.ids, tf.cast(self.prev_a, tf.int64)], axis=1))[:,tf.newaxis,:]
        customer_node_coords = self.coords[:,1:,:]
        travel_times_to_customers = self._get_travel_time(current_node_coords, customer_node_coords)
        arrival_times_at_customers = self.current_time + travel_times_to_customers
        service_start_times = tf.maximum(arrival_times_at_customers, self.ready_time[:,1:])
        mask_due_date_violation = (service_start_times > self.due_date[:,1:])
        mask_due_date_violation = mask_due_date_violation[:,tf.newaxis,:]

        # 4. Ưu tiên các nút có READY TIME thấp nhất
        unvisited_customers = ~mask_visited_customers
        unvisited_ready_times = tf.where(unvisited_customers, self.ready_time[:,1:], tf.ones_like(self.ready_time[:,1:]) * tf.float32.max)
        min_ready_time = tf.reduce_min(unvisited_ready_times, axis=1, keepdims=True)
        mask_higher_ready_time = (self.ready_time[:,1:] > min_ready_time)[:,tf.newaxis,:]

        # Tổng hợp mask cho khách hàng
        mask_customers = mask_visited_customers | mask_exceeds_capacity | mask_due_date_violation | mask_higher_ready_time

        # --- Mask cho Depot (node 0) ---
        currently_at_depot = (self.prev_a == 0)
        all_customers_done = self.all_customers_visited_per_instance()[:, tf.newaxis]

        # Điều kiện xe cần quay về depot:
        # 1. Xe đầy
        vehicle_full = (self.used_capacity > self.VEHICLE_CAPACITY)
        # 2. Không còn khách hàng hợp lệ để thăm
        no_valid_customers = tf.reduce_all(mask_customers, axis=-1)
        # 3. Không còn khách hàng nào để thăm
        no_customers_left = all_customers_done

        # Xe được phép quay về depot nếu:
        # - Không đang ở depot VÀ
        # - (Xe đầy HOẶC không còn khách hàng hợp lệ HOẶC không còn khách hàng nào)
        can_return_to_depot = ~currently_at_depot & (vehicle_full | no_valid_customers | no_customers_left)

        # Bị cấm chọn depot nếu:
        # - Đang ở depot VÀ vẫn còn khách hàng chưa thăm
        prohibit_selecting_depot = tf.logical_and(currently_at_depot, ~all_customers_done)

        # Tạo mask cho depot
        mask_depot = tf.logical_or(prohibit_selecting_depot, ~can_return_to_depot)[:, :, None]

        final_mask = tf.concat([mask_depot, mask_customers], axis=-1)
        return final_mask


    def step(self, action):
        """
        Cập nhật trạng thái dựa trên hành động (nút được chọn).
        action: (batch_size,) int32 tensor chứa index của nút được chọn.
        """
        selected_node_idx = tf.cast(action, tf.int32)[:, tf.newaxis] # (batch_size, 1)
        batch_indices_for_gather = tf.range(self.batch_size, dtype=tf.int32)[:, tf.newaxis] # (batch_size, 1)

        # Lấy thông tin của nút hiện tại (trước khi di chuyển)
        current_node_info_indices = tf.concat([batch_indices_for_gather, tf.cast(self.prev_a, tf.int32)], axis=1)
        current_node_coords = tf.gather_nd(self.coords, current_node_info_indices)

        # Lấy thông tin của nút được chọn (nút kế tiếp)
        selected_node_info_indices = tf.concat([batch_indices_for_gather, selected_node_idx], axis=1)
        selected_coords = tf.gather_nd(self.coords, selected_node_info_indices)
        selected_demand = tf.gather_nd(self.demand, selected_node_info_indices)[:, tf.newaxis]
        selected_ready_time = tf.gather_nd(self.ready_time, selected_node_info_indices)[:, tf.newaxis]
        selected_due_date = tf.gather_nd(self.due_date, selected_node_info_indices)[:, tf.newaxis]
        selected_service_time = tf.gather_nd(self.service_time, selected_node_info_indices)[:, tf.newaxis]

        is_selected_depot = (selected_node_idx == 0)

        # 1. Cập nhật thời gian hiện tại của xe
        travel_time = self._get_travel_time(current_node_coords, selected_coords)[:, tf.newaxis] # (batch_size, 1)
        arrival_at_selected = self.current_time + travel_time
        service_start_time_at_selected = tf.maximum(arrival_at_selected, selected_ready_time)
        departure_time_from_selected = service_start_time_at_selected + selected_service_time

        # Cập nhật thời gian hiện tại
        self.current_time = tf.where(is_selected_depot,
                                    tf.zeros_like(self.current_time), # Reset thời gian khi quay về depot
                                    departure_time_from_selected)

        # 2. Cập nhật prev_a (nút hiện tại của xe)
        self.prev_a = selected_node_idx

        # 3. Cập nhật used_capacity
        self.used_capacity = tf.where(is_selected_depot,
                                    tf.zeros_like(self.used_capacity), # Reset capacity khi về depot
                                    self.used_capacity + selected_demand)

        # 4. Cập nhật visited_customers_overall
        is_customer_node = ~is_selected_depot # True nếu nút được chọn là khách hàng
        customer_idx_to_update = selected_node_idx - 1 # (0 đến n_loc-1)

        # Chỉ thực hiện update nếu nút được chọn là khách hàng hợp lệ
        can_update_visit = tf.logical_and(is_customer_node, customer_idx_to_update >= 0)
        can_update_visit_sq = tf.squeeze(can_update_visit, axis=-1) # (batch_size,)

        # Lấy các chỉ số của các instance và khách hàng cần cập nhật
        valid_batch_indices = tf.where(can_update_visit_sq) # (num_valid_updates, 1)
        if tf.size(valid_batch_indices) > 0:
            actual_customer_indices = tf.gather_nd(customer_idx_to_update, valid_batch_indices) # (num_valid_updates, 1)
            # Tạo scatter_indices (batch_idx, 0_for_dim1, customer_idx)
            scatter_indices = tf.concat([
                tf.cast(valid_batch_indices, tf.int32),
                tf.zeros_like(actual_customer_indices, dtype=tf.int32), # dim_1_index is 0
                tf.cast(actual_customer_indices, tf.int32)
            ], axis=1)
            updates = tf.ones(tf.shape(scatter_indices)[0], dtype=tf.uint8) # giá trị 1 để đánh dấu đã thăm
            self.visited_customers_overall = tf.tensor_scatter_nd_update(
                self.visited_customers_overall,
                scatter_indices,
                updates
            )

        # 5. Cập nhật số lượng xe đã điều phối (vehicles_dispatched_count)
        # Tăng khi xe quay về depot VÀ vẫn còn khách hàng chưa được thăm VÀ số xe hiện tại < MAX_VEHICLES_K
        returned_to_depot_and_not_all_done = tf.logical_and(
            tf.squeeze(is_selected_depot, axis=-1),
            ~self.all_customers_visited_per_instance()
        )
        can_dispatch_new_vehicle = self.vehicles_dispatched_count < self.MAX_VEHICLES_K

        increment_vehicle_trigger = tf.logical_and(returned_to_depot_and_not_all_done, tf.squeeze(can_dispatch_new_vehicle, axis=-1))

        self.vehicles_dispatched_count = tf.where(
            increment_vehicle_trigger[:, tf.newaxis],
            self.vehicles_dispatched_count + 1,
            self.vehicles_dispatched_count
        )

        # 6. Tăng bộ đếm bước chung
        self.i = self.i + 1

    @staticmethod
    def get_costs(dataset_input_tuple, pi, alpha=ALPHA_PENALTY, beta=BETA_PENALTY, speed=SPEED, max_vehicles_k_from_problem=MAX_VEHICLES_K):
        # ... (Phần này bạn đã có, đảm bảo nó hoạt động với pi kiểu int32)
        # Lưu ý: dataset_input_tuple là dữ liệu gốc, còn pi là lộ trình được tạo ra.
        # Logic tính chi phí cần chính xác theo định nghĩa VRPTW của bạn.
        # (Giữ nguyên phần get_costs bạn đã cung cấp)
        depot_coords_batch = dataset_input_tuple[0]
        depot_time_windows_batch = dataset_input_tuple[1] # [ready, due, service] của depot
        customer_coords_batch = dataset_input_tuple[2]
        # customer_demands_batch = dataset_input_tuple[3] # Không dùng trực tiếp trong cost này nhưng cần cho capacity
        customer_ready_times_batch = dataset_input_tuple[4]
        customer_due_dates_batch = dataset_input_tuple[5]
        customer_service_times_batch = dataset_input_tuple[6]

        coords_full = tf.concat((depot_coords_batch[:, None, :], customer_coords_batch), axis=1)

        depot_ready_val = depot_time_windows_batch[:, 0:1]
        depot_due_val = depot_time_windows_batch[:, 1:2]
        depot_service_val = depot_time_windows_batch[:, 2:3]

        ready_time_full = tf.concat([depot_ready_val, customer_ready_times_batch], axis=1)
        due_date_full = tf.concat([depot_due_val, customer_due_dates_batch], axis=1)
        service_time_full = tf.concat([depot_service_val, customer_service_times_batch], axis=1)

        batch_size = tf.shape(pi)[0] # pi nên là int32

        total_travel_distance_agg = tf.zeros(batch_size, dtype=tf.float32)
        total_penalty_agg = tf.zeros(batch_size, dtype=tf.float32)
        num_vehicles_actually_used_batch = tf.zeros(batch_size, dtype=tf.int32)


        # Vòng lặp for b_idx in tf.range(batch_size) rất không hiệu quả trong graph mode.
        # Cần vector hóa logic này nếu có thể, hoặc sử dụng tf.vectorized_map / tf.while_loop.
        # Dưới đây là logic giả định cho một instance (cần điều chỉnh cho batch processing)
        # Để đơn giản, tạm thời giữ nguyên vòng lặp của bạn, nhưng lưu ý về hiệu suất.
        for b_idx in tf.range(batch_size):
            pi_b = pi[b_idx]
            actual_pi_b_indices = pi_b[pi_b >= 0] # Giả sử pi chứa index, >=0 là hợp lệ, -1 là padding

            if tf.size(actual_pi_b_indices) == 0:
                continue

            current_vehicle_tour_nodes_indices = [] # Lưu các node index của tour hiện tại
            vehicle_count_for_instance = 0
            current_time_for_vehicle_cost_calc = depot_ready_val[b_idx,0] # Bắt đầu từ thời gian sẵn sàng của depot
            current_pos_coords_for_vehicle_cost_calc = depot_coords_batch[b_idx]

            for node_idx_in_pi in actual_pi_b_indices: # Lặp qua các nút trong lộ trình của instance b
                is_depot_node = (node_idx_in_pi == 0)

                # Nếu gặp depot và tour hiện tại có nút -> kết thúc tour, bắt đầu xe mới
                if is_depot_node:
                    if not current_vehicle_tour_nodes_indices: # Bỏ qua depot ở đầu nếu là nút đầu tiên và tour rỗng
                        current_time_for_vehicle_cost_calc = depot_ready_val[b_idx,0] # Reset thời gian cho xe mới
                        current_pos_coords_for_vehicle_cost_calc = depot_coords_batch[b_idx]
                        continue

                    # Xe quay về depot để kết thúc tour hiện tại
                    if vehicle_count_for_instance < max_vehicles_k_from_problem :
                        dist_to_depot = tf.norm(current_pos_coords_for_vehicle_cost_calc - depot_coords_batch[b_idx])
                        total_travel_distance_agg = tf.tensor_scatter_nd_add(total_travel_distance_agg, [[b_idx]], [dist_to_depot])

                        arrival_at_depot = current_time_for_vehicle_cost_calc + dist_to_depot / speed
                        penalty_at_depot = beta * tf.maximum(0.0, arrival_at_depot - due_date_full[b_idx, 0])
                        total_penalty_agg = tf.tensor_scatter_nd_add(total_penalty_agg, [[b_idx]], [penalty_at_depot])
                        # Không cộng service time của depot vào current_time ở đây vì xe kết thúc.

                    # Reset cho xe mới/tour mới
                    current_vehicle_tour_nodes_indices = []
                    vehicle_count_for_instance += 1
                    if vehicle_count_for_instance < max_vehicles_k_from_problem: # Chỉ tăng số xe sử dụng nếu nó thực sự được dùng
                        num_vehicles_actually_used_batch = tf.tensor_scatter_nd_add(num_vehicles_actually_used_batch, [[b_idx]], [1])

                    current_time_for_vehicle_cost_calc = depot_ready_val[b_idx,0] # Reset thời gian cho xe mới
                    current_pos_coords_for_vehicle_cost_calc = depot_coords_batch[b_idx]

                    if vehicle_count_for_instance >= max_vehicles_k_from_problem:
                        break # Đã dùng hết K xe cho instance này
                    continue # Chuyển sang nút tiếp theo trong pi (nếu có)

                # Nếu là nút khách hàng
                if vehicle_count_for_instance >= max_vehicles_k_from_problem:
                    continue # Bỏ qua nếu đã dùng hết K xe

                # Đây là lần đầu tiên xe này rời depot trong tour này
                if not current_vehicle_tour_nodes_indices and vehicle_count_for_instance == 0 :
                     num_vehicles_actually_used_batch = tf.tensor_scatter_nd_add(num_vehicles_actually_used_batch, [[b_idx]], [1])


                current_vehicle_tour_nodes_indices.append(node_idx_in_pi)
                target_node_coords = coords_full[b_idx, node_idx_in_pi]
                dist_to_target = tf.norm(target_node_coords - current_pos_coords_for_vehicle_cost_calc)
                total_travel_distance_agg = tf.tensor_scatter_nd_add(total_travel_distance_agg, [[b_idx]], [dist_to_target])

                arrival_at_target = current_time_for_vehicle_cost_calc + dist_to_target / speed
                ready_target = ready_time_full[b_idx, node_idx_in_pi]
                due_target = due_date_full[b_idx, node_idx_in_pi]
                service_target = service_time_full[b_idx, node_idx_in_pi]

                penalty_early = alpha * tf.maximum(0.0, ready_target - arrival_at_target)
                penalty_late = beta * tf.maximum(0.0, arrival_at_target - due_target)
                total_penalty_agg = tf.tensor_scatter_nd_add(total_penalty_agg, [[b_idx]], [penalty_early + penalty_late])

                service_start_actual = tf.maximum(arrival_at_target, ready_target)
                current_time_for_vehicle_cost_calc = service_start_actual + service_target
                current_pos_coords_for_vehicle_cost_calc = target_node_coords

            # Xử lý tour cuối cùng nếu pi không kết thúc bằng depot VÀ xe cuối cùng chưa về depot
            if current_vehicle_tour_nodes_indices and vehicle_count_for_instance < max_vehicles_k_from_problem:
                dist_to_depot_final_tour = tf.norm(current_pos_coords_for_vehicle_cost_calc - depot_coords_batch[b_idx])
                total_travel_distance_agg = tf.tensor_scatter_nd_add(total_travel_distance_agg, [[b_idx]], [dist_to_depot_final_tour])
                arrival_at_depot_final_tour = current_time_for_vehicle_cost_calc + dist_to_depot_final_tour / speed
                penalty_at_depot_final_tour = beta * tf.maximum(0.0, arrival_at_depot_final_tour - due_date_full[b_idx, 0])
                total_penalty_agg = tf.tensor_scatter_nd_add(total_penalty_agg, [[b_idx]], [penalty_at_depot_final_tour])
                # Số xe sử dụng đã được tính khi xe rời depot lần đầu hoặc khi một tour kết thúc và tour mới bắt đầu.

        final_total_cost = total_travel_distance_agg + total_penalty_agg
        return final_total_cost