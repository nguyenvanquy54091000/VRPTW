import tensorflow as tf
from scipy.stats import ttest_rel
from tqdm import tqdm
import numpy as np
from attention_model import AttentionModel, set_decode_type
from generate_data import load_data_from_csv

def copy_of_tf_model(model, train_csv_path, normalize_capacity=None, n_encode_layers=None, path=None):
    ds_full = load_data_from_csv(train_csv_path, normalize_capacity=normalize_capacity)
    for sample_batch in ds_full.batch(2):
        _ = model(sample_batch, training=False) # Gọi với training=False/None
        break

    # 2) Instantiate a new model with same architecture
    embedding_dim = getattr(model, 'embedding_dim', 128)
    if n_encode_layers is None:
        # fallback to attribute on model or default
        n_encode_layers = getattr(model, 'n_encode_layers', 3)
    model_loaded = AttentionModel(embedding_dim, n_encode_layers=n_encode_layers)
    set_decode_type(model_loaded, "greedy")

    # Build new model by running same batch
    for sample_batch in ds_full.batch(2):
        _ = model_loaded(sample_batch, training=False) # Gọi với training=False/None
        break

    # 3) Copy weights via TensorFlow Checkpoint
    ckpt_orig = tf.train.Checkpoint(model=model)
    ckpt_path = ckpt_orig.write('temp_baseline_ckpt')
    ckpt_new = tf.train.Checkpoint(model=model_loaded)
    ckpt_new.restore(ckpt_path).expect_partial()

    # 4) If user provided separate checkpoint, override
    if path is not None:
        model_loaded.load_weights(path)

    return model_loaded


def rollout(model, dataset, batch_size=1000, disable_tqdm=False):
    # Evaluate model in greedy mode
    set_decode_type(model, "sampling")
    costs_list = []
    for batch in tqdm(dataset.batch(batch_size), disable=disable_tqdm, desc="Rollout greedy execution"):
        cost, _ = model(batch, training=False) # Gọi với training=False/None
        costs_list.append(cost)
    return tf.concat(costs_list, axis=0)


def validate(dataset, model, batch_size=1000):
    """Validates model on given dataset in greedy mode"""
    val_costs = rollout(model, dataset, batch_size=batch_size)
    set_decode_type(model, "sampling")
    mean_cost = tf.reduce_mean(val_costs)
    print(f"Validation score: {np.round(mean_cost, 4)}")
    return mean_cost



class RolloutBaseline:
    def __init__(self, model, filename, from_checkpoint=False,
                 path_to_checkpoint=None, wp_n_epochs=1, epoch=0,
                 num_samples=10000, warmup_exp_beta=0.8,
                 embedding_dim=128, graph_size=20,
                 train_csv_path=None, validation_csv_path=None,
                 normalize_capacity=None):
        self.model = model
        self.filename = filename
        self.from_checkpoint = from_checkpoint
        self.path_to_checkpoint = path_to_checkpoint
        self.wp_n_epochs = wp_n_epochs
        self.cur_epoch = epoch
        self.num_samples = num_samples
        self.warmup_exp_beta = warmup_exp_beta
        self.embedding_dim = embedding_dim
        self.graph_size = graph_size
        self.train_csv_path = train_csv_path
        self.validation_csv_path = validation_csv_path
        self.normalize_capacity = normalize_capacity
        self.alpha = 0.0
        self.mean = None
        self.bl_vals = None

    def _update_baseline(self, model, epoch):
        # Load or copy baseline model
        if self.from_checkpoint and self.alpha == 0:
            print('Baseline model loaded from checkpoint')
            self.model = copy_of_tf_model(
                model,
                train_csv_path=self.train_csv_path,
                normalize_capacity=self.normalize_capacity,
                n_encode_layers=self.embedding_dim,
                path=self.path_to_checkpoint
            )
        else:
            self.model = copy_of_tf_model(
                model,
                train_csv_path=self.train_csv_path,
                normalize_capacity=self.normalize_capacity,
                n_encode_layers=self.embedding_dim
            )

        # Load dataset from CSV for baseline estimation
        if self.train_csv_path is None:
            raise ValueError("train_csv_path must be provided to RolloutBaseline")
        self.dataset = load_data_from_csv(self.train_csv_path, normalize_capacity=self.normalize_capacity)

        print(f"Evaluating baseline model on baseline dataset (epoch = {epoch})")
        self.bl_vals = rollout(self.model, self.dataset)
        self.mean = tf.reduce_mean(self.bl_vals)
        self.cur_epoch = epoch

    def ema_eval(self, cost):
        """Running average of cost through previous batches (only warm-up)"""
        if self.mean is None:
            return cost
        v_b = self.mean
        v_ema = cost if self.alpha == 0 else (self.alpha * v_b + (1 - self.alpha) * cost)
        return v_ema

    def eval_all(self, dataset=None):
        """Evaluates current baseline model on the whole validation CSV (non warm-up) or given dataset"""
        if self.alpha < 1:
            return None
        # optionally override with validation CSV
        if self.validation_csv_path is not None:
            dataset = load_data_from_csv(self.validation_csv_path, normalize_capacity=self.normalize_capacity)
        if dataset is None:
            raise ValueError("No dataset provided for eval_all")
        val_costs = rollout(self.model, dataset, batch_size=2048)
        return val_costs

    def epoch_callback(self, model, epoch):
        self.cur_epoch = epoch
        print(f"Evaluating candidate model on baseline dataset (callback epoch = {self.cur_epoch})")
        candidate_vals = rollout(model, self.dataset)
        candidate_mean = tf.reduce_mean(candidate_vals)
        diff = candidate_mean - self.mean
        print(f"Epoch {self.cur_epoch} candidate mean {candidate_mean}, baseline mean {self.mean}, difference {diff}")
        if diff < 0:
            t, p = ttest_rel(candidate_vals, self.bl_vals)
            p_val = p / 2
            print(f"p-value: {p_val}")
            if p_val < 0.05:
                print('Update baseline')
                self._update_baseline(model, self.cur_epoch)
        if self.alpha < 1.0:
            self.alpha = (self.cur_epoch + 1) / float(self.wp_n_epochs)
