# layers.py
from __future__ import print_function
import tensorflow as tf
import numpy as np


class MultiHeadAttention(tf.keras.layers.Layer):
    # ... (docstring và __init__ giữ nguyên) ...
    def __init__(self, n_heads, d_model, **kwargs):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_depth = self.d_model // self.n_heads

        if self.d_model % self.n_heads != 0:
            raise ValueError("number of heads must divide d_model")

        self.wq = tf.keras.layers.Dense(self.d_model, use_bias=False)
        self.wk = tf.keras.layers.Dense(self.d_model, use_bias=False)
        self.wv = tf.keras.layers.Dense(self.d_model, use_bias=False)
        self.w_out = tf.keras.layers.Dense(self.d_model, use_bias=False)

    def split_heads(self, tensor, batch_size):
        tensor = tf.reshape(tensor, (batch_size, -1, self.n_heads, self.head_depth))
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]

        Q = self.wq(q)
        K = self.wk(k)
        V = self.wv(v)

        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        compatibility = tf.matmul(Q, K, transpose_b=True)
        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        compatibility = compatibility / tf.math.sqrt(dk)

        if mask is not None:
            # Đảm bảo mask là Tensor trước khi cố gắng reshape và sử dụng
            if not tf.is_tensor(mask):
                print(f"[WARNING] MultiHeadAttention.call received non-Tensor mask (type: {type(mask)}). Value: {mask}. Masking will be skipped.")
            else:
                # Mask mong đợi có shape (batch_size, seq_len_q, seq_len_k)
                # Reshape mask để broadcast qua các heads:
                # (batch_size, seq_len_q, seq_len_k) --> (batch_size, 1, seq_len_q, seq_len_k)
                reshaped_mask_for_broadcast = mask[:, tf.newaxis, :, :]

                # Áp dụng mask. tf.where yêu cầu mask là boolean.
                # Giả định rằng mask đầu vào đã là boolean hoặc có thể cast an toàn.
                # Nếu mask đầu vào (trước khi reshape) có giá trị True ở những vị trí cần được che (masked out).
                compatibility = tf.where(
                    tf.cast(reshaped_mask_for_broadcast, tf.bool), # Đảm bảo mask là boolean
                    tf.ones_like(compatibility) * (-np.inf),
                    compatibility
                )

        compatibility = tf.nn.softmax(compatibility, axis=-1)
        attention = tf.matmul(compatibility, V)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        attention = tf.reshape(attention, (batch_size, -1, self.d_model))
        output = self.w_out(attention)

        return output