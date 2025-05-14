import tensorflow as tf
from time import gmtime, strftime
from attention_model import AttentionModel, set_decode_type
from reinforce_baseline import RolloutBaseline
from train import train_model_from_csv
from generate_data import load_data_from_csv, get_cur_time, get_journey_khach

TRAIN_CSV_PATH      = r"E:\Innter\VRPTW\Data\R201.csv"
VALIDATION_CSV_PATH = r"E:\Innter\VRPTW\Data\R201.csv"
NORMALIZE_CAPACITY  = 200.0
BATCH_SIZE        = 512
VAL_BATCH_SIZE    = 10000
START_EPOCH       = 0
END_EPOCH         = 40
FROM_CHECKPOINT   = False
EMBEDDING_DIM     = 512
LEARNING_RATE     = 1e-4
ROLLOUT_SAMPLES   = 10000
WP_N_EPOCHS       = 1
GRAD_NORM_CLIP    = 1.0
GRAPH_SIZE        = 100
FILENAME          = f"VRP_{GRAPH_SIZE}_{strftime('%Y-%m-%d', gmtime())}"

# --- Model & optimizer ---
model_tf = AttentionModel(EMBEDDING_DIM)
set_decode_type(model_tf, "sampling")
print(get_cur_time(), "Model initialized")

optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

# --- Baseline ---
baseline = RolloutBaseline(
    model_tf,
    filename=FILENAME,
    from_checkpoint=FROM_CHECKPOINT,
    wp_n_epochs=WP_N_EPOCHS,
    epoch=START_EPOCH,
    num_samples=ROLLOUT_SAMPLES,
    embedding_dim=EMBEDDING_DIM,
    graph_size=GRAPH_SIZE,
    train_csv_path=TRAIN_CSV_PATH,
    validation_csv_path=VALIDATION_CSV_PATH,
    normalize_capacity=NORMALIZE_CAPACITY
)
print(get_cur_time(), "Baseline initialized")

# --- Train ---
train_model_from_csv(
    optimizer,
    model_tf,
    baseline,
    train_csv_path=TRAIN_CSV_PATH,
    validation_csv_path=VALIDATION_CSV_PATH,
    batch=BATCH_SIZE,
    val_batch_size=VAL_BATCH_SIZE,
    start_epoch=START_EPOCH,
    end_epoch=END_EPOCH,
    grad_norm_clipping=GRAD_NORM_CLIP,
    filename=FILENAME,
    normalize_capacity=NORMALIZE_CAPACITY,
)

# --- Hiển thị một tuyến đường mẫu trên tập validation ---
val_data = load_data_from_csv(VALIDATION_CSV_PATH, normalize_capacity=NORMALIZE_CAPACITY)
print(get_cur_time(), "Hiển thị tuyến đường mẫu:")
for batch in val_data.batch(1):
    cost, ll, pi = model_tf(batch, return_pi=True)
    get_journey_khach(batch, pi, ind_in_batch=0)
    break
