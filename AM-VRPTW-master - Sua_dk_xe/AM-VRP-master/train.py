from tqdm import tqdm
import pandas as pd
import tensorflow as tf
from attention_model import set_decode_type
from reinforce_baseline import validate, RolloutBaseline
from generate_data import generate_data_onfly, get_results, get_cur_time, load_data_from_csv, get_journey_khach
from time import gmtime, strftime

def train_model_from_csv(optimizer,
                model_tf,
                baseline: RolloutBaseline,
                train_csv_path,
                validation_csv_path,
                batch = 128,    
                val_batch_size = 1000,
                start_epoch = 0,
                end_epoch = 5,
                grad_norm_clipping = 1.0,
                filename = None,
                normalize_capacity=None,
                ):
    train_data = load_data_from_csv(train_csv_path, normalize_capacity=normalize_capacity)
    val_data = load_data_from_csv(validation_csv_path, normalize_capacity=normalize_capacity)
    train_loss_results = []
    train_cost_results = []
    val_cost_avg = []

    # Training loop
    for epoch in range(start_epoch, end_epoch):
        print(f"Starting epoch {epoch}")
        baseline._update_baseline(model_tf, epoch)
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_cost_avg = tf.keras.metrics.Mean()
        for num_batch, x_batch in tqdm(enumerate(train_data.batch(batch)), desc=f"Epoch {epoch}"):
            with tf.GradientTape() as tape:
                cost, log_likelihood = model_tf(x_batch, training = True)
                bl_val = baseline.ema_eval(cost)
                reinforce_loss = tf.reduce_mean((cost - tf.stop_gradient(bl_val)) * log_likelihood)
            grads = tape.gradient(reinforce_loss, model_tf.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, grad_norm_clipping)
            optimizer.apply_gradients(zip(grads, model_tf.trainable_variables))
            epoch_loss_avg.update_state(reinforce_loss)
            epoch_cost_avg.update_state(cost)
        baseline.epoch_callback(model_tf, epoch)
        set_decode_type(model_tf, "sampling")
        val_score = validate(val_data, model_tf, batch_size=val_batch_size)
        print(f"Epoch {epoch} val cost: {val_score}")
        train_loss_results.append(epoch_loss_avg.result().numpy())
        train_cost_results.append(epoch_cost_avg.result().numpy())
        val_cost_avg.append(val_score.numpy())


    set_decode_type(model_tf, "greedy")

    print("Determining best route on validation set...")
    best_cost = float('inf')
    best_batch = None
    best_pi = None
    best_idx = None
    for batch_val in val_data.batch(val_batch_size):
        cost_vals, _, pi = model_tf(batch_val, return_pi=True)
        cost_vals = cost_vals.numpy()
        idx = cost_vals.argmin()
        if cost_vals[idx] < best_cost:
            best_cost = cost_vals[idx]
            best_batch = batch_val
            best_pi = pi  # keep full batch pi tensor
            best_idx = idx
    print(f"Lowest cost found: {best_cost}")
    # display route for the best instance in best_batch
    get_journey_khach(best_batch, best_pi, ind_in_batch=best_idx)

    # save results and plots
    get_results(train_loss_results, train_cost_results, val_cost_avg,
                save_results=True, filename=filename, plots=True)
