import pickle
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import time


def create_data_on_disk(graph_size, num_samples, is_save=True, filename=None, is_return=False, seed=1234):
    CAPACITIES = {
        10: 20.,
        20: 30.,
        50: 40.,
        100: 50.
    }
    depo, graphs, demand = (tf.random.uniform(minval=0, maxval=1, shape=(num_samples, 2), seed=seed),
                            tf.random.uniform(minval=0, maxval=1, shape=(num_samples, graph_size, 2), seed=seed),
                            tf.cast(tf.random.uniform(minval=1, maxval=10, shape=(num_samples, graph_size),
                                                      dtype=tf.int32, seed=seed), tf.float32) / tf.cast(CAPACITIES[graph_size], tf.float32)
                            )
    
    if is_save:
        save_to_pickle('Validation_dataset_{}.pkl'.format(filename), (depo, graphs, demand))
    if is_return:
        return tf.data.Dataset.from_tensor_slices((list(depo), list(graphs), list(demand)))


def save_to_pickle(filename, item):
    with open(filename, 'wb') as handle:
        pickle.dump(item, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_from_pickle(path, return_tf_data_set=True, num_samples=None):
    objects = []
    with (open(path, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    objects = objects[0]
    if return_tf_data_set:
        depo, graphs, demand = objects
        if num_samples is not None:
            return tf.data.Dataset.from_tensor_slices((list(depo), list(graphs), list(demand))).take(num_samples)
        else:
            return tf.data.Dataset.from_tensor_slices((list(depo), list(graphs), list(demand)))
    else:
        return objects


def generate_data_onfly(num_samples=10000, graph_size=20):
    CAPACITIES = {
        10: 20.,
        20: 30.,
        50: 40.,
        100: 50.
    }
    
    depo, graphs, demand = (tf.random.uniform(minval=0, maxval=1, shape=(num_samples, 2)),
                            tf.random.uniform(minval=0, maxval=1, shape=(num_samples, graph_size, 2)),
                            tf.cast(tf.random.uniform(minval=1, maxval=10, shape=(num_samples, graph_size),
                                                      dtype=tf.int32), tf.float32)/tf.cast(CAPACITIES[graph_size], tf.float32)
                            )
    # Đặt demand của depo thành 0
    # depo_demand = tf.zeros((num_samples, 1), dtype=tf.float32)
    # demand = tf.concat([depo_demand, demand], axis=1)
    return tf.data.Dataset.from_tensor_slices((list(depo), list(graphs), list(demand)))

# def load_data_from_csv(path, normalize_capacity=None):
#     df = pd.read_csv(path)
#     depo = df.iloc[0][['XCOORD.', 'YCOORD.']].to_numpy(dtype='float32')
#     cust = df.iloc[1:]
#     graphs = cust[['XCOORD.', 'YCOORD.']].to_numpy(dtype='float32')
#     demand = cust['DEMAND'].to_numpy(dtype='float32')
#     if normalize_capacity is not None:
#         demand = demand / float(normalize_capacity)
#     # Chuyển dữ liệu sang danh sách
#     depo_list = [depo]  # Danh sách chứa tọa độ của depo
#     graphs_list = [graphs]  # Danh sách chứa tọa độ của các điểm khách hàng
#     demand_list = [np.concatenate([demand])]  # Danh sách chứa demand (bao gồm cả depo)
#     return tf.data.Dataset.from_tensor_slices((depo_list, graphs_list, demand_list))
def load_data_from_csv(path, normalize_capacity=None):
    df = pd.read_csv(path)

    # Thông tin Depot (dòng đầu tiên)
    depot_row = df.iloc[0]
    depot_coords_val = depot_row[['XCOORD.', 'YCOORD.']].to_numpy(dtype='float32')
    # Thông tin thời gian của Depot
    depot_ready_time_val = np.array([depot_row['READY TIME']], dtype='float32')
    depot_due_date_val = np.array([depot_row['DUE DATE']], dtype='float32')
    depot_service_time_val = np.array([depot_row['SERVICE TIME']], dtype='float32')
    # Gộp thông tin thời gian của depot thành một mảng (hoặc tensor)
    depot_time_windows_val = np.concatenate([depot_ready_time_val, depot_due_date_val, depot_service_time_val])


    # Thông tin Khách hàng (các dòng còn lại)
    cust_df = df.iloc[1:]
    customer_coords_val = cust_df[['XCOORD.', 'YCOORD.']].to_numpy(dtype='float32')
    customer_demands_val = cust_df['DEMAND'].to_numpy(dtype='float32')
    customer_ready_times_val = cust_df['READY TIME'].to_numpy(dtype='float32')
    customer_due_dates_val = cust_df['DUE DATE'].to_numpy(dtype='float32')
    customer_service_times_val = cust_df['SERVICE TIME'].to_numpy(dtype='float32')

    if normalize_capacity is not None:
        customer_demands_val = customer_demands_val / float(normalize_capacity)
        # Cân nhắc chuẩn hóa cả thời gian nếu cần, nhưng thường không cần thiết nếu các đơn vị nhất quán.

    # Tạo danh sách cho tf.data.Dataset (mỗi instance là một bài toán)
    # Vì file CSV chỉ chứa một instance, ta sẽ tạo danh sách chứa một phần tử.
    depot_coords_list = [depot_coords_val]
    depot_time_windows_list = [depot_time_windows_val] # List chứa array [ready, due, service] của depot
    customer_coords_list = [customer_coords_val]
    customer_demands_list = [customer_demands_val]
    customer_ready_times_list = [customer_ready_times_val]
    customer_due_dates_list = [customer_due_dates_val]
    customer_service_times_list = [customer_service_times_val]

    # Trả về tf.data.Dataset với 7 thành phần
    return tf.data.Dataset.from_tensor_slices((
        depot_coords_list,
        depot_time_windows_list, # Mới: thông tin thời gian của depot
        customer_coords_list,
        customer_demands_list,
        customer_ready_times_list,
        customer_due_dates_list,
        customer_service_times_list
    ))


def get_results(train_loss_results, train_cost_results, val_cost, save_results=True, filename=None, plots=True):
    epochs_num = len(train_loss_results)
    df_train = pd.DataFrame(data={'epochs': list(range(epochs_num)),
                                  'loss': train_loss_results,
                                  'cost': train_cost_results,
                                  })
    df_test = pd.DataFrame(data={'epochs': list(range(epochs_num)),
                                 'val_сost': val_cost})
    if save_results:
        df_train.to_excel('train_results_{}.xlsx'.format(filename), index=False)
        df_test.to_excel('test_results_{}.xlsx'.format(filename), index=False)

    if plots:
        plt.figure(figsize=(15, 9))
        ax = sns.lineplot(x='epochs', y='loss', data=df_train, color='salmon', label='train loss')
        ax2 = ax.twinx()
        sns.lineplot(x='epochs', y='cost', data=df_train, color='cornflowerblue', label='train cost', ax=ax2)
        sns.lineplot(x='epochs', y='val_сost', data=df_test, palette='darkblue', label='val cost').set(ylabel='cost')
        ax.legend(loc=(0.75, 0.90), ncol=1)
        ax2.legend(loc=(0.75, 0.95), ncol=2)
        ax.grid(axis='x')
        ax2.grid(True)
        plt.savefig('learning_curve_plot_{}.jpg'.format(filename))
        plt.show()


def get_journey(batch, pi, ind_in_batch=0):
    pi_ = get_clean_path(pi[ind_in_batch].numpy())
    depo_coord = batch[0][ind_in_batch].numpy()
    points_coords = batch[1][ind_in_batch].numpy()
    demands = batch[2][ind_in_batch].numpy()
    node_labels = ['(' + str(x[0]) + ', ' + x[1] + ')' for x in enumerate(demands.round(2).astype(str))]
    full_coords = np.concatenate((depo_coord.reshape(1, 2), points_coords))
    list_of_paths = []
    cur_path = []
    for idx, node in enumerate(pi_):

        cur_path.append(node)

        if idx != 0 and node == 0:
            if cur_path[0] != 0:
                cur_path.insert(0, 0)
            list_of_paths.append(cur_path)
            cur_path = []

    list_of_path_traces = []
    for path_counter, path in enumerate(list_of_paths):
        coords = full_coords[[int(x) for x in path]]
        lengths = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
        total_length = np.sum(lengths)
        list_of_path_traces.append(go.Scatter(x=coords[:, 0],
                                              y=coords[:, 1],
                                              mode="markers+lines",
                                              name=f"path_{path_counter}, length={total_length:.2f}",
                                              opacity=0.5))

    trace_points = go.Scatter(x=points_coords[:, 0],
                              y=points_coords[:, 1],
                              mode='markers+text',
                              name='destinations',
                              text=node_labels,
                              textposition='top center',
                              opacity=0.5
                              )

    trace_depo = go.Scatter(x=[depo_coord[0]],
                            y=[depo_coord[1]],
                            text=['1.0'], textposition='bottom center',
                            mode='markers+text',
                            name='depo'
                            )

    layout = go.Layout(title='<b>Пример работы модели</b>',
                       xaxis=dict(title='Y coordinate'),
                       yaxis=dict(title='X coordinate'),
                       showlegend=True,
                       width=800,
                       height=700
                       )

    data = [trace_points, trace_depo] + list_of_path_traces
    print(pi_)
    fig = go.Figure(data=data, layout=layout)
    fig.show()

def get_journey_khach(batch, pi, ind_in_batch=0):
    # Làm sạch chuỗi hành trình
    pi_ = get_clean_path(pi[ind_in_batch].numpy())
    
    # Lấy thông tin từ batch
    depo_coord = batch[0][ind_in_batch].numpy()
    points_coords = batch[2][ind_in_batch].numpy()  # Thay đổi từ batch[1] sang batch[2]
    demands = batch[3][ind_in_batch].numpy()  # Thay đổi từ batch[2] sang batch[3]
    
    # Kiểm tra và xử lý points_coords
    if points_coords.ndim == 1:
        if len(points_coords) == 2:
            points_coords = points_coords.reshape(1, 2)
        elif len(points_coords) == 3:
            points_coords = points_coords[:2].reshape(1, 2)
        else:
            raise ValueError(f"points_coords có kích thước không hợp lệ: {points_coords.shape}")
    
    # Gán nhãn cho các điểm
    node_labels = ['(' + str(x[0]) + ', ' + x[1] + ')' for x in enumerate(demands.round(2).astype(str))]
    
    # Kết hợp tọa độ depo và các điểm
    full_coords = np.concatenate((depo_coord.reshape(1, 2), points_coords), axis=0)
    
    # Tạo danh sách các tuyến đường
    list_of_paths = []
    cur_path = []
    for idx, node in enumerate(pi_):
        cur_path.append(node)
        if idx != 0 and node == 0:  # Quay lại depo
            if cur_path[0] != 0:
                cur_path.insert(0, 0)  # Đảm bảo tuyến đường bắt đầu từ depo
            list_of_paths.append(cur_path)
            cur_path = []

    # Vẽ các tuyến đường
    list_of_path_traces = []
    for path_counter, path in enumerate(list_of_paths):
        try:
            # Kiểm tra và chuyển đổi các index trong path
            valid_indices = [int(x) for x in path if 0 <= int(x) < len(full_coords)]
            if not valid_indices:
                print(f"Warning: Path {path_counter} không có index hợp lệ")
                continue
                
            coords = full_coords[valid_indices]
            
            # Tính độ dài của tuyến đường
            if len(coords) > 1:
                lengths = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
                total_length = np.sum(lengths)
            else:
                total_length = 0
            
            # Thêm tuyến đường vào danh sách vẽ
            list_of_path_traces.append(go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers+lines",
                name=f"Tuyến {path_counter}, chiều dài={total_length:.2f}",
                opacity=0.7
            ))
        except Exception as e:
            print(f"Warning: Không thể vẽ tuyến đường {path_counter}: {str(e)}")
            continue

    # Vẽ các điểm khách hàng
    trace_points = go.Scatter(
        x=points_coords[:, 0],
        y=points_coords[:, 1],
        mode='markers+text',
        name='Khách hàng',
        text=node_labels,
        textposition='top center',
        marker=dict(size=8, color='blue'),
        opacity=0.8
    )

    # Vẽ depo
    trace_depo = go.Scatter(
        x=[depo_coord[0]],
        y=[depo_coord[1]],
        mode='markers+text',
        name='Depo',
        text=['Depo'],
        textposition='bottom center',
        marker=dict(size=10, color='red'),
        opacity=1.0
    )

    # Cấu hình biểu đồ
    layout = go.Layout(
        title='<b>Biểu đồ các tuyến đường thăm khách</b>',
        xaxis=dict(title='X coordinate'),
        yaxis=dict(title='Y coordinate'),
        showlegend=True,
        width=800,
        height=700
    )

    # Tạo và hiển thị biểu đồ
    data = [trace_points, trace_depo] + list_of_path_traces
    fig = go.Figure(data=data, layout=layout)
    fig.show()


def get_cur_time():
    ts = time.time()
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')


def get_clean_path(arr):
    p1, p2 = 0, 1
    output = []
    while p2 < len(arr):
        if arr[p1] != arr[p2]:
            output.append(arr[p1])
            if p2 == len(arr) - 1:
                output.append(arr[p2])
        p1 += 1
        p2 += 1
    if output[0] != 0:
        output.insert(0, 0.0)
    if output[-1] != 0:
        output.append(0.0)
    return output

