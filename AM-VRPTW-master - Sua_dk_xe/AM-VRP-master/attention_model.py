# attention_model.py
import tensorflow as tf
from attention_graph_encoder import GraphAttentionEncoder
from attention_graph_decoder import GraphAttentionDecoder
from Enviroment import VRPproblem # Đảm bảo import này đúng

# Hàm set_decode_type gốc của bạn (nếu vẫn cần)
def set_decode_type(model, decode_type):
    model.set_decode_type(decode_type) # Giả sử model có phương thức này
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'set_decode_type'):
        model.decoder.set_decode_type(decode_type)


class AttentionModel(tf.keras.Model):
    def __init__(self,
                 embedding_dim,
                 n_encode_layers=3,
                 n_heads=8,
                 tanh_clipping=10.
                 ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None # Sẽ được set bởi set_decode_type
        self.problem = VRPproblem # Đảm bảo VRPproblem được định nghĩa đúng
        self.n_heads = n_heads

        self.embedder = GraphAttentionEncoder(input_dim=self.embedding_dim,
                                              num_heads=self.n_heads,
                                              num_layers=self.n_encode_layers
                                              )
        self.decoder = GraphAttentionDecoder(num_heads=self.n_heads,
                                             output_dim=self.embedding_dim,
                                             tanh_clipping=tanh_clipping,
                                             decode_type=self.decode_type) # decode_type sẽ được cập nhật

    # Ghi đè set_decode_type nếu bạn muốn nó là một phương thức của lớp này
    def set_decode_type(self, decode_type):
        self.decode_type = decode_type
        if hasattr(self.decoder, 'set_decode_type'):
            self.decoder.set_decode_type(decode_type)

    def _calc_log_likelihood(self, _log_p, a):
        """
        Calculate log likelihood of actions.
        Args:
            _log_p: shape (batch_size, seq_len, 1, n_nodes)
            a: shape (batch_size, seq_len)
        Returns:
            log_likelihood: shape (batch_size,)
        """
        # Print shapes and types for debugging
        # print("_log_p shape in _calc_log_likelihood:", tf.shape(_log_p))
        # print("a shape in _calc_log_likelihood:", tf.shape(a))
        # print("a dtype:", a.dtype)
        
        # Convert a to int32 if it's not already
        a = tf.cast(a, tf.int32)
        # print("a dtype after cast:", a.dtype)
        
        # Reshape _log_p to (batch_size, seq_len, n_nodes)
        _log_p = tf.squeeze(_log_p, axis=2)
        # print("_log_p shape after squeeze:", tf.shape(_log_p))
        
        # Create indices for gather_nd
        batch_size = tf.shape(a)[0]
        seq_len = tf.shape(a)[1]
        
        # Create batch indices
        batch_idx = tf.range(batch_size, dtype=tf.int32)[:, tf.newaxis]
        batch_idx = tf.tile(batch_idx, [1, seq_len])
        
        # Create sequence indices
        seq_idx = tf.range(seq_len, dtype=tf.int32)[tf.newaxis, :]
        seq_idx = tf.tile(seq_idx, [batch_size, 1])
        
        # Stack indices - ensure all tensors are int32
        indices = tf.stack([batch_idx, seq_idx, a], axis=-1)
        # print("indices shape:", tf.shape(indices))
        # print("indices dtype:", indices.dtype)
        
        # Gather log probabilities
        log_p = tf.gather_nd(_log_p, indices)
        # print("log_p shape after gather_nd:", tf.shape(log_p))
        
        # Sum over sequence length
        return tf.reduce_sum(log_p, axis=1)

    def call(self, inputs, return_pi=False, training=None): # Thêm training
        # Truyền `training` cho embedder
        embeddings, mean_graph_emb = self.embedder(inputs, training=training)

        # Truyền `training` cho decoder nếu nó cũng có các lớp như BatchNormalization
        # Hiện tại GraphAttentionDecoder không có BN, nhưng thêm vào không hại.
        _log_p, pi = self.decoder(inputs=inputs, embeddings=embeddings, context_vectors=mean_graph_emb, training=training)


        # Đảm bảo pi được truyền cho get_costs là kiểu dữ liệu đúng (ví dụ: int32)
        cost = self.problem.get_costs(inputs, tf.cast(pi, tf.int32))


        ll = self._calc_log_likelihood(_log_p, pi)

        if return_pi:
            return cost, ll, pi
        return cost, ll

    # Thêm phương thức build để có thể giải quyết warning
    def build(self, input_shape):
        # input_shape ở đây là một tuple các shape, ví dụ:
        # ((None, 2), (None, 3), (None, None, 2), ...)
        # Chúng ta có thể gọi self.embedder.build và self.decoder.build nếu cần thiết
        # Hoặc chỉ cần gọi super().build
        # self.embedder.build(input_shape) # Gọi nếu GraphAttentionEncoder có build tường minh
        super().build(input_shape)
        self.built = True # Đánh dấu là đã build