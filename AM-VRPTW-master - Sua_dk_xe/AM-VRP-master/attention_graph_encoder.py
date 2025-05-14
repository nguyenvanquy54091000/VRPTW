import tensorflow as tf
from layers import MultiHeadAttention

class MultiHeadAttentionLayer(tf.keras.layers.Layer):
    """Feed-Forward Sublayer: fully-connected Feed-Forward network,
    built based on MHA vectors from MultiHeadAttention layer with skip-connections
        Args:
            num_heads: number of attention heads in MHA layers.
            input_dim: embedding size that will be used as d_model in MHA layers.
            feed_forward_hidden: number of neuron units in each FF layer.
        Call arguments:
            x: batch of shape (batch_size, n_nodes, node_embedding_size).
            mask: mask for MHA layer
        Returns:
               outputs of shape (batch_size, n_nodes, input_dim)
    """
    def __init__(self, input_dim, num_heads, feed_forward_hidden=512, **kwargs):
        super().__init__(**kwargs)
        self.mha = MultiHeadAttention(n_heads=num_heads, d_model=input_dim, name='MHA')
        self.bn1 = tf.keras.layers.BatchNormalization(name='bn1', trainable=True)
        self.bn2 = tf.keras.layers.BatchNormalization(name='bn2', trainable=True)
        self.ff1 = tf.keras.layers.Dense(feed_forward_hidden, name='ff1')
        self.ff2 = tf.keras.layers.Dense(input_dim, name='ff2')

    def call(self, x, mask=None, training=None):  # Thêm training cho BatchNormalization
        actual_mask_for_mha = mask
        if mask is not None and not tf.is_tensor(mask):
            print(f"[WARNING] MultiHeadAttentionLayer received non-Tensor mask (type: {type(mask)}). Value: {mask}. Forcing to None.")
            actual_mask_for_mha = None

        mha_out = self.mha(x, x, x, actual_mask_for_mha)
        sc1_out = x + mha_out
        bn1_out = self.bn1(sc1_out, training=training)
        ff1_out = self.ff1(bn1_out)
        relu1_out = tf.keras.activations.relu(ff1_out)
        ff2_out = self.ff2(relu1_out)
        sc2_out = bn1_out + ff2_out
        bn2_out = self.bn2(sc2_out, training=training)
        return bn2_out

class GraphAttentionEncoder(tf.keras.layers.Layer):
    def __init__(self, input_dim, num_heads, num_layers, feed_forward_hidden=512):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.feed_forward_hidden = feed_forward_hidden
        self.init_embed_depot = tf.keras.layers.Dense(self.input_dim, name='init_embed_depot')
        self.init_embed = tf.keras.layers.Dense(self.input_dim, name='init_embed')
        self.mha_layers = [MultiHeadAttentionLayer(self.input_dim, self.num_heads, self.feed_forward_hidden)
                            for _ in range(self.num_layers)]

    def call(self, x_tuple, training=None, mask=None):
        # Unpack features for depot and customers
        depot_coords_input = x_tuple[0]
        depot_time_windows_input = x_tuple[1]
        customer_coords_input = x_tuple[2]
        customer_demands_input = x_tuple[3]
        customer_ready_times_input = x_tuple[4]
        customer_due_dates_input = x_tuple[5]
        customer_service_times_input = x_tuple[6]

        # Embed depot features: [coords_x, coords_y, demand=0, ready, due, service]
        batch_size = tf.shape(depot_coords_input)[0]
        depot_demand = tf.zeros((batch_size, 1), dtype=customer_demands_input.dtype)
        depot_features = tf.concat((depot_coords_input, depot_demand, depot_time_windows_input), axis=-1)
        embedded_depot = self.init_embed_depot(depot_features)
        embedded_depot = embedded_depot[:, None, :]

        # Embed customer features: [coords_x, coords_y, demand, ready, due, service]
        customer_features = tf.concat((
            customer_coords_input,
            customer_demands_input[:, :, None],
            customer_ready_times_input[:, :, None],
            customer_due_dates_input[:, :, None],
            customer_service_times_input[:, :, None]
        ), axis=-1)
        embedded_customers = self.init_embed(customer_features)
        x_embedded = tf.concat((embedded_depot, embedded_customers), axis=1)

        # Process mask for MHA layers
        actual_mask_for_mha_layers = None
        if mask is not None:
            if isinstance(mask, tuple):
                if all(m is None for m in mask):
                    actual_mask_for_mha_layers = None
                else:
                    mask_values = [m if m is not None else tf.zeros_like(x_embedded[:, :1]) for m in mask]
                    actual_mask_for_mha_layers = tf.stack(mask_values, axis=0)
            elif tf.is_tensor(mask):
                actual_mask_for_mha_layers = mask
            else:
                print(f"[WARNING] GraphAttentionEncoder received invalid mask type: {type(mask)}. Value: {mask}. Using None.")

        # Apply multi-head attention layers
        for i in range(self.num_layers):
            x_embedded = self.mha_layers[i](x_embedded, mask=actual_mask_for_mha_layers, training=training)
        output = (x_embedded, tf.reduce_mean(x_embedded, axis=1))
        return output
