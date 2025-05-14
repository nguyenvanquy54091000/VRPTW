import tensorflow as tf
import numpy as np
from Enviroment import VRPproblem

class GraphAttentionDecoder(tf.keras.layers.Layer):
    def __init__(self,
                 output_dim,
                 num_heads,
                 tanh_clipping=10,
                 decode_type=None):
        super().__init__()
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_depth = self.output_dim // self.num_heads
        self.dk_mha_decoder = tf.cast(self.head_depth, tf.float32)
        self.dk_get_loc_p = tf.cast(self.output_dim, tf.float32)

        if self.output_dim % self.num_heads != 0:
            raise ValueError("number of heads must divide d_model=output_dim")

        self.tanh_clipping = tanh_clipping
        self.decode_type = decode_type

        # projection layers
        self.wq_context      = tf.keras.layers.Dense(self.output_dim, use_bias=False, name='wq_context')
        self.wq_step_context = tf.keras.layers.Dense(self.output_dim, use_bias=False, name='wq_step_context')
        self.wk              = tf.keras.layers.Dense(self.output_dim, use_bias=False, name='wk')
        self.wk_tanh         = tf.keras.layers.Dense(self.output_dim, use_bias=False, name='wk_tanh')
        self.wv              = tf.keras.layers.Dense(self.output_dim, use_bias=False, name='wv')
        self.w_out           = tf.keras.layers.Dense(self.output_dim, use_bias=False, name='w_out')

    def build(self, input_shape):
        super().build(input_shape)

    def set_decode_type(self, decode_type):
        self.decode_type = decode_type

    def split_heads(self, tensor, batch_size):
        """
        Split tensor into multiple heads.
        Args:
            tensor: shape (batch_size, seq_len, d_model)
            batch_size: batch size
        Returns:
            tensor: shape (batch_size, num_heads, seq_len, head_depth)
        """
        # Đảm bảo tensor có đúng 3 chiều
        tensor_shape = tf.shape(tensor)
        if len(tensor_shape) != 3:
            raise ValueError(f"Expected tensor to have 3 dimensions, got {len(tensor_shape)}")
        
        # Reshape để tách thành các heads
        tensor = tf.reshape(tensor, (batch_size, -1, self.num_heads, self.head_depth))
        
        # Transpose để có shape (batch_size, num_heads, seq_len, head_depth)
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def _select_node(self, logits):
        """
        Select next node based on logits.
        Args:
            logits: shape (batch_size, 1, 1, n_nodes) or (batch_size, 1, n_nodes)
        Returns:
            selected: shape (batch_size,)
        """
        # Print original shape for debugging
        # print("Original logits shape:", tf.shape(logits))
        
        # Reshape logits to (batch_size, n_nodes)
        # First remove all singleton dimensions
        logits_reshaped = tf.squeeze(logits)
        # print("After first squeeze:", tf.shape(logits_reshaped))
        
        # If still not 2D, add batch dimension
        if len(tf.shape(logits_reshaped)) == 1:
            logits_reshaped = tf.expand_dims(logits_reshaped, axis=0)
        # print("Final logits shape:", tf.shape(logits_reshaped))
        
        if self.decode_type == "greedy":
            selected = tf.math.argmax(logits_reshaped, axis=-1)
        elif self.decode_type == "sampling":
            selected = tf.random.categorical(logits_reshaped, 1)
            selected = tf.squeeze(selected, axis=-1)
        else:
            raise ValueError(f"Unknown decode type {self.decode_type}")
        
        # print("Selected shape:", tf.shape(selected))
        return selected

    def get_step_context(self, state, embeddings):
        prev_node = state.prev_a  # (batch, 1)
        cur_emb   = tf.gather(embeddings, tf.cast(prev_node, tf.int32), batch_dims=1)
        step_ctx  = tf.concat([cur_emb, VRPproblem.VEHICLE_CAPACITY - state.used_capacity[:, :, None]], axis=-1)
        return step_ctx  # (batch, 1, input_dim+1)

    def decoder_mha(self, Q, K, V, mask=None):
        """
        Multi-head attention for decoder.
        Args:
            Q: shape (batch_size, num_heads, 1, head_depth)
            K: shape (batch_size, num_heads, n_nodes, head_depth)
            V: shape (batch_size, num_heads, n_nodes, head_depth)
            mask: shape (batch_size, 1, n_nodes)
        Returns:
            output: shape (batch_size, 1, d_model)
        """
        batch_size = tf.shape(Q)[0]
        
        # Tính toán attention
        compatibility = tf.matmul(Q, K, transpose_b=True) / self.dk_mha_decoder
        if mask is not None:
            mask_ = mask[:, tf.newaxis, :, :]
            compatibility = tf.where(mask_, tf.ones_like(compatibility) * (-np.inf), compatibility)
        
        weights = tf.nn.softmax(compatibility, axis=-1)
        attn_vec = tf.matmul(weights, V)
        
        # Reshape attn_vec từ 5D về 4D nếu cần
        attn_vec_shape = tf.shape(attn_vec)
        if len(attn_vec_shape) == 5:
            # Reshape từ [batch, 1, heads, seq_len, head_depth] thành [batch, heads, seq_len, head_depth]
            attn_vec = tf.squeeze(attn_vec, axis=1)
        
        # Transpose và reshape
        attn_vec = tf.transpose(attn_vec, perm=[0, 2, 1, 3])  # (batch_size, seq_len, num_heads, head_depth)
        attn_vec = tf.reshape(attn_vec, (batch_size, 1, self.output_dim))  # (batch_size, 1, output_dim)
        
        return self.w_out(attn_vec)

    def get_log_p(self, Q, K_tanh, mask=None):
        """
        Calculate log probabilities for node selection.
        Args:
            Q: shape (batch_size, 1, d_model)
            K_tanh: shape (batch_size, n_nodes, d_model)
            mask: shape (batch_size, 1, n_nodes)
        Returns:
            log_p: shape (batch_size, 1, n_nodes)
        """
        # Ensure Q has correct shape
        if len(tf.shape(Q)) == 2:
            Q = tf.expand_dims(Q, axis=1)
        
        compatibility = tf.matmul(Q, K_tanh, transpose_b=True) / tf.math.sqrt(self.dk_get_loc_p)
        compatibility = tf.math.tanh(compatibility) * self.tanh_clipping
        
        if mask is not None:
            # Ensure mask has correct shape
            if len(tf.shape(mask)) == 2:
                mask = tf.expand_dims(mask, axis=1)
            compatibility = tf.where(mask, tf.ones_like(compatibility) * (-np.inf), compatibility)
        
        # Print shapes for debugging
        # print("Q shape in get_log_p:", tf.shape(Q))
        # print("K_tanh shape in get_log_p:", tf.shape(K_tanh))
        # print("Compatibility shape:", tf.shape(compatibility))
        
        return tf.nn.log_softmax(compatibility, axis=-1)

    def call(self, inputs, embeddings, context_vectors, return_pi=False, training=None):
        batch_size = tf.shape(embeddings)[0]
        state = VRPproblem(inputs)

        # Precompute projections
        K = self.wk(embeddings)
        K_tanh = self.wk_tanh(embeddings)
        V = self.wv(embeddings)
        Q_context = self.wq_context(context_vectors[:, None, :])

        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        outputs = []
        sequences = []
        step_count = 0
        
        # Tăng giới hạn số bước decode
        max_steps = state.n_loc * 4 + state.MAX_VEHICLES_K * 4  # Tăng hệ số từ 2 lên 4
        
        # Thêm biến để theo dõi tiến trình
        last_progress = 0
        progress_check_interval = max_steps // 10  # Kiểm tra mỗi 10% tiến trình

        while not state.all_finished() and step_count < max_steps:
            step_context = self.get_step_context(state, embeddings)
            Q_step = self.wq_step_context(step_context)
            Q = Q_context + Q_step
            Q_split = self.split_heads(Q, batch_size)

            mask = state.get_mask()
            mha_output = self.decoder_mha(Q_split, K, V, mask)
            log_p = self.get_log_p(mha_output, K_tanh, mask)
            selected = self._select_node(log_p)

            # Clamp selected indices to valid range
            n_nodes = tf.shape(embeddings)[1]
            max_idx = tf.cast(n_nodes - 1, selected.dtype)
            min_idx = tf.cast(0, selected.dtype)
            selected = tf.clip_by_value(selected, min_idx, max_idx)

            state.step(selected)
            outputs.append(log_p[:, 0, :])
            sequences.append(selected)
            step_count += 1

            # Kiểm tra và báo cáo tiến trình
            if step_count - last_progress >= progress_check_interval:
                print(f"Decoding progress: {step_count}/{max_steps} steps")
                last_progress = step_count

        # if step_count >= max_steps:
        #     print(f"Warning: reached max_decode_steps = {max_steps}. Consider increasing max_steps if this happens frequently.")

        log_probs = tf.stack(outputs, axis=1)
        pi_tensor = tf.cast(tf.stack(sequences, axis=1), tf.int32)

        if return_pi:
            return log_probs, pi_tensor
        return log_probs, tf.cast(pi_tensor, tf.float32)
